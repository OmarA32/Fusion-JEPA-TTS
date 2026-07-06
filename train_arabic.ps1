$ErrorActionPreference = 'Stop'
cd Audio-JEPA
..\venv\Scripts\python src\train.py data=arabic_tts trainer.accelerator=cpu ++trainer.max_epochs=1 ++trainer.limit_train_batches=2 ++trainer.limit_val_batches=0 ++trainer.limit_test_batches=0 tags=[test]
cd ..
