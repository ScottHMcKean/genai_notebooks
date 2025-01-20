# Databricks notebook source
# MAGIC %pip install --upgrade azure-ai-formrecognizer azure-ai-ml azure-ai-documentintelligence azure-search-documents
# MAGIC %restart_python

# COMMAND ----------

import json
import os
from dataclasses import dataclass

from IPython.display import Markdown as md
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from azure.ai.formrecognizer import AnalyzeResult as FormRecognizerAnalyzeResult
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    AnalyzeDocumentRequest,
    DocumentAnalysisFeature,
)

# ignore cryptography version warnings
import warnings
warnings.filterwarnings(action='ignore', module='.*cryptography.*')

# Append src module to system path to import from src module
import sys
sys.path.append(os.path.abspath("../function_app"))

from src.components.doc_intelligence import (
    DefaultDocumentPageProcessor, DefaultDocumentKeyValuePairProcessor,
    DefaultDocumentTableProcessor, DefaultDocumentFigureProcessor,
    DefaultDocumentParagraphProcessor, DefaultDocumentLineProcessor,
    DefaultDocumentWordProcessor, DefaultSelectionMarkFormatter,
    DefaultDocumentSectionProcessor, DocumentIntelligenceProcessor, 
    PageDocumentListSplitter, convert_processed_di_docs_to_openai_message,
    convert_processed_di_docs_to_markdown,
)
from src.helpers.data_loading import load_pymupdf_pdf, extract_pdf_page_images

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Display all outputs of a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# COMMAND ----------


