from pyspark.sql.types import *

cust_witsml_schema = StructType([
    StructField('wellSet', StructType([
        StructField('well', StructType([
            StructField('name', StringType(), True),
            StructField('uid', StringType(), True),
            StructField('wellboreSet', StructType([
                StructField('wellbore', StructType([
                    StructField('name', StringType(), True),
                    StructField('uid', StringType(), True),
                    StructField('dtsInstalledSystemSet', StructType([
                        StructField('dtsInstalledSystem', StructType([
                            StructField('name', StringType(), True),
                            StructField('uid', StringType(), True),
                            StructField('fiberInformation', StructType([
                                StructField('fiber', StructType([
                                    StructField('name', StringType(), True),
                                    StructField('uid', StringType(), True),
                                    StructField('mode', StringType(), True)
                                ]), True)
                            ]), True)
                        ]), True)
                    ]), True),
                    StructField('dtsMeasurementSet', StructType([
                        StructField('dtsMeasurement', StructType([
                            StructField('name', StringType(), True),
                            StructField('uid', StringType(), True),
                            StructField('installedSystemUsed', StringType(), True),
                            StructField('dataInWellLog', StringType(), True),
                            StructField('connectedToFiber', StringType(), True)
                        ]), True)
                    ]), True),
                    StructField('wellLogSet', StructType([
                        StructField('wellLog', StructType([
                            StructField('name', StringType(), True),
                            StructField('uid', StringType(), True),
                            StructField('serviceCompany', StringType(), True),
                            StructField('creationDate', TimestampType(), True),
                            StructField('indexType', StringType(), True),
                            StructField('logCurveInfo', ArrayType(StructType([
                                StructField('mnemonic', StringType(), True),
                                StructField('unit', StringType(), True),
                                StructField('curveDescription', StringType(), True),
                                StructField('classWitsml', StringType(), True),
                                StructField('uid', StringType(), True)
                            ])), True),
                            StructField('blockInfo', StructType([
                                StructField('indexType', StringType(), True),
                                StructField('direction', StringType(), True),
                                StructField('indexCurve', StringType(), True),
                                StructField('uid', StringType(), True),
                                StructField('blockCurveInfo', ArrayType(StructType([
                                    StructField('curveId', StringType(), True),
                                    StructField('columnIndex', IntegerType(), True)
                                ])), True)
                            ]), True),
                            StructField('logData', StructType([
                                StructField('data', ArrayType(StringType()), True)
                            ]), True)
                        ]), True)
                    ]), True)
                ]), True)
            ]), True)
        ]), True)
    ]), True)
])