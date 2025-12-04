FROM quay.io/jupyter/minimal-notebook:2025-11-24

USER root

RUN sudo apt update \
    && sudo apt install -y lmodern \
    && sudo apt install -y gdebi-core
    
RUN curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-amd64.deb \
	&& sudo gdebi --non-interactive quarto-1.5.57-linux-amd64.deb \
	&& rm quarto-1.5.57-linux-amd64.deb

USER $NB_UID

COPY conda-lock.yml /tmp/conda-lock.yml
RUN mamba install --quiet --file /tmp/conda-lock.yml \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
RUN pip install deepchecks==0.19.1
RUN pip install pandera==0.27.0



