FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev

RUN mkdir Workspace
WORKDIR Workspace/
COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR Notebook
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8899", "--no-browser", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]