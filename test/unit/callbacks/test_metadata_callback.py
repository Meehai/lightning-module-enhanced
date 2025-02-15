from tempfile import TemporaryDirectory
import shutil
from pathlib import Path
from lightning_module_enhanced.callbacks import MetadataCallback

def test_metadata_callback_save_and_load_equallity(request):
    # this also tests that keys are preserved as is (i.e epoch metric from int to str)
    (log_dir := Path(TemporaryDirectory().name)).mkdir(exist_ok=False)
    request.addfinalizer(lambda: shutil.rmtree(log_dir, ignore_errors=True))
    cb = MetadataCallback()
    cb.metadata = {"epoch_metrics": {"loss": {0: [1.2], 1: [5.55]}}}
    cb.log_dir = log_dir
    cb.log_file_path = f"{cb.log_dir}/metadata.json"
    cb.save()

    cb2 = MetadataCallback()
    cb2.load_state_dict(open(cb.log_dir / "metadata.json", "r").read())

    assert cb.metadata == cb2.metadata
