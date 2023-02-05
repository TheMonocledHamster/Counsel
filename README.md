# Proximal Policy Optimization (Clip) on a [GCN](https://arxiv.org/abs/1609.02907v4)-embedded Actor-Critic Model for Dynamic Resource Configuration Management on the Cloud.

[![DOI](https://zenodo.org/badge/553064046.svg)](https://zenodo.org/badge/latestdoi/553064046)

Descriptive steps to reproduce COUNSEL:

1.  Clone repository from Github or Zenodo.\
    [https://github.com/TheMonocledHamster/Counsel\
    ](https://github.com/TheMonocledHamster/Counsel)<https://zenodo.org/badge/latestdoi/553064046>

2.  Ensure Docker Engine and Docker Compose are installed.\
    [https://docs.docker.com/engine/install/\
    ](https://docs.docker.com/engine/install/)<https://docs.docker.com/compose/install/>

3.  From the root directory of the project, (Counsel/) run docker compose up. The build process should take 5-10 minutes, and training may require around 100 hours, depending on the capability of the system.\
    (Alternatively, if the build takes too long or runs into issues, use docker pull themonocledhamster/counsel and docker pull themonocledhamster/load-gen in order to pull ready images, then proceed with docker compose up as usual.)\
    [<https://hub.docker.com/repository/docker/themonocledhamster/counsel>,\
    <https://hub.docker.com/repository/docker/themonocledhamster/load-gen>]

4.  The hyperparameters can be configured in model/configs/hyperparams.json. In case of limited workstation capabilities, threadcount and epoch length can be varied

5.  All necessary training and inference will be run within the docker containers. At any time during or after the training, the copy_data.sh script can be run to fetch the latest training output from the container.

6.  bash copy_data.sh or zsh copy_data.sh can be run to fetch output.

7.  In your python runtime environment, install pandas, matplotlib, scipy and cycler:\
    pip install matplotlib\
    pip install pandas\
    pip install scipy\
    pip install cycler\
    Ensure you are using a reasonably recent version of python (3.7+).

8.  From within the plotters folder, run the following scripts to visualize the model's results. Ideally, you should run this step from an IDE (like PyCharm) or a code editor (like VSCode) in order to render the GUI of the graphs.

1.  Figure 4: plot_prov.py

2.  Figure 5: plot_hp.py

3.  Figure 6: plot_infer_time.py

10. Partial graphs can be plotted even as the training is ongoing, although the final plot requires that training is completed.
