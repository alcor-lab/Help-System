FROM gcr.io/neon-rampart-204112/project-tools

ARG DEBIAN_FRONTEND=noninteractive
ENV REPOSITORY=Help-System
ENV BUILD_DIR="${BUILD_HOME}/${REPOSITORY}"

#get all the project dependencies
COPY --chown=999:999 . ${BUILD_DIR}
WORKDIR ${BUILD_DIR}
RUN set -x \
    && apt-get update -qq \
    && hey project setup get --dependencies \
    && hey project setup get --all \
    && chown 999:999 -R "${PYTHONUSERBASE}" \
    && chown 999:999 -R "${BUILD_DIR}" \
    && rm -rf /var/lib/apt/lists/*

#install all dependencies and examples
USER ${BUILD_USER}
RUN set -x \
    && hey project setup install --dependencies \
    && hey project setup install --examples

USER root
VOLUME ${VOLUME_DIR}
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
