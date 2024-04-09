from omegaconf import DictConfig
import hydra
import experiments.discriminative


# experiments from the VCL paper that can be carried out
EXP_OPTIONS = {
    'disc_p_mnist': experiments.discriminative.permuted_mnist,
    'disc_s_mnist': experiments.discriminative.split_mnist,
    'disc_s_n_mnist': experiments.discriminative.split_not_mnist
}


@hydra.main(config_path="config", config_name="base")
def main(config: DictConfig):
    # run all experiments
    if config.experiment == 'all':
        for exp in list(EXP_OPTIONS.keys()):
            print("Running", exp)
            EXP_OPTIONS[exp](config)
    # select specific experiment to run
    else:
        EXP_OPTIONS[config.experiment](config)


if __name__ == "__main__":
    main()