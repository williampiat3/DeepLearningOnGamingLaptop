# Deep Learning On A Gaming Laptop
This folder presents the installation that we made on a Lenovo Legion Y920 with i7 and a GPU NVIDIA GeForce 1070 in my former company, for a better agility we managed for our work in Applied Deep learning. This would give the possibility to anyone to build their own installation so that they can practice freely.
Although this guide is well written and ends up in a perfectly working setup, you will need some linux skills in order to fix the issues that will be coming after this installation (if you update the cuda drivers or other things that might eventually arrive).

This installation was made on a specific model of computer and you might have issues transfering it to another, but it gives a lot of solution to a wide range of problems we encountered. I believe it is a rather complete guide.

## Installation version
By following this guide you'll install this environement:

On a Lenovo Legion Y920 - Core i7-7820HK - 16 Go - SSD 256 Go + HDD 1 To - Screen 17.3" LED Full HD - NVIDIA GeForce GTX 1070 8 Go you'll have:

* A dual boat Ubuntu 18.04 LTS/ Windows 10
* Cuda 10.0
* Cudnn
* Python 3.7
* Tensorflow-gpu > 2.0
* Pytorch 1.4  for cuda 10.0


Along with other libraries that are quite handy for manipulating data and visulizations

## Material requirements
You'll need:
* A bootable key with Ubuntu 18.04
* A paperclip (or any pointy thing you'll find)
* Patience (seriously)
* The freshly bought PC 

## Installation

### Freeing the Hard Drive partition
To install ubuntu we will have to disable some of windows functionalities. Once you are in windows:

* Open a command line
* Type `diskmgmt.msc` and enter
* The window of the partitions of the PC opens
* Delete the partition on the Hard Drive (1 To)

Then you need to disable the "fast boot":
* Go to the parameters
* Select "Power Management"
* Clic on "Change parameters currently unavailable"
* Untick "fast boot (recommanded)"
* Clic on "Save changes"


### Installing Ubuntu

#### Installing the OS
On this computer Windows is installed on the SSD, we will therefore install Ubuntu on the Hard Drive.

Enter the BIOS of the computer (on the computer you have a button on the right side which can be pressed with a pointy tool)

On the configuration tab, change "SATA Controller mode" to "AHSI" save and quit the BIOS, the computer attemps to start on windows, turn it off

Plug the Ubuntu 18.04 bootable key 

Start using the button on the right side to enter BIOS menu, start the computer on the USB drive

Ubuntu opens and clic on install Ubuntu, select your langage.
You can decide to just wipe out Windows and install ubuntu instead on the SSD and skip to the next section but here we will choose "Something Else" with "normal installation". 

In the space we just freed on the hard drive we will create 3 partitions:

Select the 1 To space and clic on "+"

* Primary partition
* Beginning of this space (200 000 Mo)
* File ext4
* Mounting point "/"

On the remaining free space, clic on "+":
* Primary partition
* Beginning of this space (1024 Mo)
* Swap file

On the remaining free space, clic on "+":
* Primary partition
* Beginning of this space (all the remaining space)
* File ext4
* Mounting point "/home"

Device where the booting program will be installed: "NVMe device", the SSD

Clic on install now

Ubuntu will ask you some other information, for the password, select something easy to type because the installations will ask for it **an awful lot**

Reboot the computer, once the screen is black (but not off) remove the bootable key and press enter to turn off.

**Remark: if you want to start Windows you'll have to return to the BIOS and change back the SATA Controler Mode back to intel RST**

#### Configuring Ubuntu

** On this PC we noticed that the installation of Ubuntu had some unexpected results, the wifi chip, the bluetooth chip are not working at first, we suggest you use an ethernet cable but we will solve this problem right away

1. Problem with the chips
..* Check the state of the wifi chip:
..* `sudo lshw -C network`
..* `rfkill list`
If there are 2 wifi chips it means that this is the issue
..* `sudo tee /etc/modprobe.d/ideapad.conf <<< "blacklist ideapad_laptop"`
..* Reboot


2. Update Ubuntu 18.04 (a window should suggest you to update once you are connected to the web)

3. Install some dependancies:
..* `sudo apt get update`
..* `sudo apt get upgrade`
..* `sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils software-properties-common git`

#### Install CUDA 10
1. Check compatibility of the graphic card: `lspci | grep -i nvidia`
2. Some informations about the GPU should be displayed otherwise g[o here to solve the issue](https://docs.nvidia.com/cuda/archive/10.0/)
3. Install some dependancies, python3.7, pip3.7:
..* `sudo apt-get install build-essential cmake unzip zip`
..* `sudo add-apt-repository ppa:deadsnakes/ppa`
..* `sudo apt-get update`
..* `sudo apt-get install python3.7 python3.7-dev pylint`
..* `sudo python3.7 get-pip.py`
..* `sudo pip3.7 install testresources numpy matplotlib`
4. Download CUDA 10.0 at [this link](https://developer.nvidia.com/cuda-10.0-download-archive) (Linux-x86_64  -> Ubuntu-18.04  -> deb (local))




 











