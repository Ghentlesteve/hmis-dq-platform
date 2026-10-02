# Spark job image: Java 21 + Python 3.12 + PySpark + the S3A connector + this project.
#
#   docker compose build spark
#   docker compose run --rm spark hmis-dq spark smoke
#
# Layers are ordered from least to most frequently changed, so editing code
# only rebuilds the last layer instead of re-downloading 650 MB of jars.

FROM eclipse-temurin:21-jre-noble

ARG PYSPARK_VERSION=4.1.3
# hadoop-aws must match the Hadoop version bundled with PySpark (3.4.2 for 4.1.x),
# and the AWS SDK version must match what that hadoop-aws release was built against.
ARG HADOOP_AWS_VERSION=3.4.2
ARG HADOOP_AWS_SHA1=16f9de6da5c7862241d55a13d34843404d0cbb88
ARG AWS_SDK_VERSION=2.29.52
ARG AWS_SDK_SHA1=b63eb928c2ac13bde07746bcfe7fb38bbdb4b5c5

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv tini procps \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYSPARK_PYTHON=/opt/venv/bin/python
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir "pyspark==${PYSPARK_VERSION}"

# S3A connector jars, verified against Maven Central's published checksums
ARG JARS=/opt/venv/lib/python3.12/site-packages/pyspark/jars
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar ${JARS}/
ADD https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/${AWS_SDK_VERSION}/bundle-${AWS_SDK_VERSION}.jar ${JARS}/
RUN cd ${JARS} \
    && echo "${HADOOP_AWS_SHA1}  hadoop-aws-${HADOOP_AWS_VERSION}.jar" | sha1sum -c - \
    && echo "${AWS_SDK_SHA1}  bundle-${AWS_SDK_VERSION}.jar" | sha1sum -c - \
    && chmod 644 *.jar

# Project dependencies first (cached), then the source code
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[spark]"

# Don't run as root
RUN useradd --create-home --uid 1001 hmis
USER hmis

# tini forwards Ctrl+C to Spark so jobs stop cleanly
ENTRYPOINT ["tini", "--"]
CMD ["hmis-dq", "--help"]
