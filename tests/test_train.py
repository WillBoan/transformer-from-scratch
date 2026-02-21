from train import Trainer


def test_training_loop():
    trainer = Trainer(resume=False)
    trainer.train(max_iters=10)
