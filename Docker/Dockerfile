FROM pytorch/pytorch:latest

ARG UID=1000
ARG GID=1000
ARG NAME=base

# # # RUN sudo apt-key adv --keyserver keys.gnupg.net --recv-keys A4B469963BF863CC

# # # Install basic tools
# # RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update -y
RUN apt-get install sudo ssh vim python3-pip ffmpeg libsm6 libxext6 -y

# Install pip tools
RUN pip3 install scikit-image  && \
	pip3 install opencv-python  && \
	pip3 install notebook  && \
	pip3 install matplotlib  && \
	pip3 install tensorboard  

# Add User
RUN groupadd -g $GID -o $NAME

RUN useradd -u $UID -m -g $NAME -G plugdev $NAME && \ 
	echo "$NAME ALL = NOPASSWD: ALL" > /etc/sudoers.d/user && \
	chmod 0440 /etc/sudoers.d/user

RUN chown -R $NAME:$NAME /home/$NAME

USER $NAME

# ADD PATH
RUN export PATH="/home/$NAME/.local/bin:$PATH"

RUN echo "export PS1=\"\[\e[0;31m\]\u@\[\e[m\e[0;34m\]\h\[\e[m \e[0;32m\] \w[\!]\$\[\e[m\]  \"" >> ~/.bashrc

# run bash
CMD ["/bin/bash"]