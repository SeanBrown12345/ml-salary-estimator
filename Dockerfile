FROM quay.io/jupyter/minimal-notebook

COPY conda-lock.yml /tmp/conda-lock.yml
RUN mamba install --quiet --file /tmp/conda-lock.yml \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
