from benchmarl.algorithms import MappoConfig, QmixConfig, MasacConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

if __name__ == '__main__':
    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.train_device = "cuda"
    experiment = Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=exp_config
    )
    experiment.run()

    benchmark = Benchmark(
        algorithm_configs=[
            MappoConfig.get_from_yaml(),
            QmixConfig.get_from_yaml(),
            MasacConfig.get_from_yaml(),
        ],
        tasks=[
            VmasTask.BALANCE.get_from_yaml(),
            VmasTask.SAMPLING.get_from_yaml(),
        ],
        seeds={0, 1},
        experiment_config=ExperimentConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
    )
    benchmark.run_sequential()