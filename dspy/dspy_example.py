# Databricks notebook source
# DBTITLE 1,pip installs
# MAGIC %pip install -U -qqqq databricks-agents dspy-ai mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set Configuration
import yaml

with open("dspy_rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
ft_table = rag_config.get("demo_config").get("source_table")+"_raft"
serving_endpoint_name = rag_config.get("demo_config").get("endpoint_name") + "_ft"

# COMMAND ----------

# DBTITLE 1,DSPy Setup
import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the model
lm = dspy.Databricks(model="databricks-meta-llama-3-70b-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=600) # Baseline/distill 70b parameter model
ft_lm = dspy.Databricks(model=serving_endpoint_name, model_type="text", api_key=token, 
                        api_base=url + "/serving-endpoints", max_tokens=600) # Using a smaller and cheaper fine tuned model
judge = dspy.Databricks(model="databricks-dbrx-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=600) # Use different model as judge
dspy.settings.configure(lm=lm)

# COMMAND ----------

# DBTITLE 1,Set Retrieval Model
from dspy.retrieve.databricks_rm import DatabricksRM

rm = DatabricksRM(
    databricks_index_name=rag_config.get("vector_search_index"),
    databricks_endpoint=url,
    databricks_token=token,
    columns=[rag_config.get("chunk_column_name"), rag_config.get("document_source_id")],
    text_column_name=rag_config.get("chunk_column_name"),
    docs_id_column_name=rag_config.get("document_source_id"), # TODO: return doc id too, use in eval
    k=rag_config.get("vector_search_parameters").get("k")
)
rm(query="Transportation and logistic issues", query_type="text")

# COMMAND ----------

# DBTITLE 1,Define Golden Dataset
from pyspark.sql.functions import length

synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
golden_dataset = (
    spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
    .orderBy(length("expected_response").asc()) # Use smaller responses to train
    .limit(100) # Limit to 100 records for demo - in production, use 300+
).toPandas()
train_cutoff = int(len(golden_dataset) * .6)
dataset = [dspy.Example(request=row['request'], response=row['expected_response'], 
                        context=row['expected_retrieved_context']).with_inputs('request') 
           for i, row in golden_dataset.iterrows()]
trainset = dataset[:train_cutoff]
testset = dataset[train_cutoff:]

# COMMAND ----------

# DBTITLE 1,DSPy Signatures and Modules
class QueryExpand(dspy.Signature):
    """Rephrases the question to increase the quality of the response"""
    question = dspy.InputField(desc="A question about our business")
    expanded_question = dspy.OutputField(desc="A slightly expanded question that provides more context to our retrieval engine")

class QuestionAnswer(dspy.Signature):
    """Returns the answer to the question"""
    request = dspy.InputField(desc="A question about our business")
    response = dspy.OutputField(desc="A short, numbered answer to the question")

class QuestionAnswerAgent(dspy.Module):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.query_expander = dspy.ChainOfThought(QueryExpand)
        self.answer_cot = dspy.ChainOfThought(QuestionAnswer) 

    def forward(self, request): # stitch together the RAG chain
        expanded_question = self.query_expander(question=request).expanded_question
        docs = " ".join(self.rm(expanded_question).docs)
        return self.answer_cot(request=request+docs)

qa_agent = QuestionAnswerAgent(rm)
pred = qa_agent(dataset[0].request)
pred

# COMMAND ----------

class AssessResponse(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    request = dspy.InputField()
    response_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

# DBTITLE 1,DSPy Metric
# Define the metric for optimization
def metric(gold, pred, trace=None):
    request, expected_response = gold.request, gold.response # , gold.expected_retrieved_context
    response = pred.response
    with dspy.context(lm=judge):
        specific_q = "Does the response provide a very detailed, specific response to the request?"
        value_add_q = "Does the response avoid simply repeating back the provided request and add value to the conversation?"
        not_duplicative_q = "Are the items in the list of responses unique rather than repetitive?"
        listed_q = "Is the response formatted in a bulleted or numbered list?"
        specific_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=specific_q)
        value_add_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=value_add_q)
        not_duplicative_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=not_duplicative_q)
        listed_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=listed_q)
        target_len = 600
        max_len_score = 3 # Our users asked for short, numbered lists, so we'll incentivize that behavior 
        length_eval = max((target_len - len(response)) / (target_len/max_len_score), 0) 

        evals = [not_duplicative_eval, listed_eval]
        results = ['yes' in m.assessment_answer.lower() for m in evals]
        score = (sum(results) + length_eval) / (len(evals) + max_len_score) # Total score over total possible score
        if trace is not None: # When training, we'll only return positive signal at a high score
            # print(results, length_eval, evals)
            return score > .5
        return score

# COMMAND ----------

# DBTITLE 1,DSPy Evaluation Function
def evaluate(testset, system):
    scores = []
    results = []
    for x in testset:
        pred = system(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
        results.append(pred)
    return sum(scores) / len(testset), results

# COMMAND ----------

# DBTITLE 1,Run DSPy Evaluate
with dspy.context(lm=ft_lm): 
    baseline_ft_score, baseline_ft_results = evaluate(testset, qa_agent)
print(f"Baseline fine tune score:    {baseline_ft_score * 100:.2f}%")

# COMMAND ----------

# DBTITLE 1,MLflow Evaluation Function
import mlflow
import pandas as pd

def mlflow_evaluate(testset, results):
    eval_output = []
    for i, prediction in enumerate(results):
        eval_output.append({
            "request": testset[i].request,
            "response": prediction.response,
            "expected_response": testset[i].response
        })
    eval_metrics = mlflow.evaluate(
        data=pd.DataFrame(eval_output),
        model_type="databricks-agent"
    ).metrics
    return eval_metrics
    
mlflow_evaluate(testset, baseline_ft_results)

# COMMAND ----------

# DBTITLE 1,DSPy Optimization
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(prompt_model=lm, task_model=ft_lm, metric=metric, num_candidates=3, init_temperature=0.1)

# Optimize the program
kwargs = dict(num_threads=4, display_progress=True, display_table=0)
with dspy.context(lm=ft_lm):
    optimized_qa_agent = optimizer.compile(
        student=qa_agent,
        trainset=trainset, 
        eval_kwargs=kwargs,
        requires_permission_to_run=False,
    )

# COMMAND ----------

# DBTITLE 1,DSPy Evaluation
# Evaluate the optimized program
optimized_score, optimized_results = evaluate(testset, optimized_qa_agent)
print(f"Optimized fine tune score:    {optimized_score * 100:.2f}%")

# COMMAND ----------

# DBTITLE 1,MLflow Evaluation
mlflow_evaluate(testset, optimized_results)
