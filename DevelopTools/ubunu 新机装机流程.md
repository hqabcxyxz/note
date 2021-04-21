```bash
sudo apt-get install python
sudo apt-get install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py
sudo apt-get install python3
sudo apt-get install python3-pip
# fzf 略
# proxychains
mkdir hq_config
cd !$
git clone https://github.com/rofl0r/proxychains-ng.git
cd proxychains-ng/
./configure --prefix=/usr --sysconfdir=/etc
sudo make && make install
sudo make install-config
cd ../

# cuda
# 下载run文件之后
sudo bash cuda.run
# accept,可以只用选toolkit
# 设置.bashrc
vim ~/.bashrc
#加入如下内容
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
#以下内容移除path中重复路径
export PATH=$( python -c "import os; path = os.environ['PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export LD_LIBRARY_PATH=$( python -c "import os; path = os.environ['LD_LIBRARY_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export LIBRARY_PATH=$( python -c "import os; path = os.environ['LIBRARY_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export PKG_CONFIG_PATH=$( python -c "import os; path = os.environ['PKG_CONFIG_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )

# cudnn
tar xvf cudnn.tar
sudo cp cuda/include/cudnn.h    /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*

# anaconda
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# download anaconda
bash anaconda.sh
#anzhuan

# rg
sudo apt-get install ripgrep

# change .bashrc
export FZF_TMUX=1
export FZF_TMUX_OPTS='-p 80%'
export FZF_DEFAULT_COMMAND='rg --color always --files --no-ignore --hidden --follow -g "!{.git,node_modules}/*" 2> /dev/null'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_DEFAULT_OPTS=" --tiebreak=index --ansi --border --preview '(highlight -O ansi {} || cat {}) 2> /dev/null | head -500'"

# lazygit
sudo add-apt-repository ppa:lazygit-team/release
sudo apt-get update
sudo apt-get install lazygit

# ranger
git clone https://github.com/ranger/ranger.git
cd ranger
sudo python setup.py install --optimize=1 --record=install_log.txt

# 安装配置
git clone https://github.com/captainfffsama/LinuxConfig.git

# z 
git clone https://github.com/rupa/z.git
# 修改.bashrc

# npm
sudo apt-get install nodejs
sudo apt-get install npm
sudo npm install npm -g
sudo npm cache clean -f 
sudo npm install -g n 
sudo n stable 
```

nvim 安装参见 [nvim环境搭建](../DevelopTools/nvim%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md)

```bash
sudo pip install pynvim
sudo pip3 install pynvim

sudo apt install xclip

# ctags

sudo apt install autoconf
cd /tmp
git clone https://github.com/universal-ctags/ctags
cd ctags
sudo apt install \
    gcc make \
    pkg-config autoconf automake \
    python3-docutils \
    libseccomp-dev \
    libjansson-dev \
    libyaml-dev \
    libxml2-dev
./autogen.sh
./configure --prefix=/opt/software/universal-ctags  # 安装路径可以况调整。
make -j8
sudo make install
```

gtags  安装见 [ubuntu  编译安装gtags](ubuntu%20%20编译安装gtags.md)