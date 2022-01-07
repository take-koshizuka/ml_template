# ML template

## Setup

1. Install Docker.
   * Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.
   * Setup running [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Clone this repository and `cd` into it.
    ```bash
    git clone https://github.com/take-koshizuka/continual-VC.git
    ```
3. Build the Docker image using Docker/Dockerfile
    ```bash
    cd Docker && docker build . -t tmp_image
    ```
4. Run the container with `-it` flag. **All subsequent steps should be executed on the container.**
    ``` bash
    docker run -v $HOME/Documents:/mnt --name tmp_env -it tmp_image
    ```

## Training

```bash
python3 train.py [-c] [-d] 
```


## Testing

```bash
python3 eval.py [-d] [-o]
```

