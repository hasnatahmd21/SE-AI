from sebrain.training_data import TrainingDatasetExporter
from sebrain.c34 import FabricRecord

def test_content_hash_duplicates_are_not_split_across_sets():
    a=FabricRecord(record_id="D01-1",dataset_id="D01",question="q",answer="a",content_hash="same",raw={"execution_status":"EXECUTED","validation_status":"VERIFIED"})
    b=FabricRecord(record_id="D01-2",dataset_id="D01",question="q",answer="a",content_hash="same",raw={"execution_status":"EXECUTED","validation_status":"VERIFIED"})
    rows=TrainingDatasetExporter().convert([a,b])
    assert len(rows)==1
