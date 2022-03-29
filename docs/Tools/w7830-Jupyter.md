> First Update: 29, March, 2022

## Introduction

You have setup the Python environment on your server, and you are about to access your Jupyter Notebook or Jupyter Lab on that. Wait what, you cannot access your Jupyter since is behind the *sshgate* or *firewall*. This tutorial will tell you how to create an `ssh` tunnel to a remote machine, and connect to this server from a browser running on your local machine to create and use jupyter notebook [^1].

## Topology

Here is what we want these ports to be forwarded:

![Topo](../../_media/topo.png)

Let me explain:

1. Server is behind the SSHGate, which is not accessible from outside, you need to use VPN and ssh it through SSHGate
2. SSHGate is also invisible from outside, you should use a VPN
3. The ultimate goal is that you can access the Jupyter Notebook on your own local machine through a browser

## Implementation

### Jupyter Server Setup

1. Install jupyter notebook by typing on your environment
   ```
   pip install jupyter
   ```
2. If you have not set up the password for your jupyter server, create on first:
   ```
   jupyter notebook --generate-config
   ```
	1. open `python`, then type in:
	   ```
	   from notebook.auth import passwd  
	   passwd()  
	   ## You should type in your password twice
	   ```
	2. cope the generate key to your the config file `~/.jupyter/jupyter_notebook_config.py`, and change these settings:
	   ```                       
	   c.NotebookApp.password = u'sha:ce...' # The one you copied
	   c.NotebookApp.open_browser = False  
	   c.NotebookApp.port =8889 # **set up the port**
	   ``` 
3. Save the config.
4. In case you don't want to shut down the connection to server, I strongly recommend you using `screen`. 
	1. type `screen` in your cmd, it will open a session for you
	2.  open jupyter server
   ```
   jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
   ```
   3. You can shut down your ssh terminal to server, since screen will run your server in the backend.
   4. If you want to access your jupyter notebook again, type `screen -list` to check your session. You can use `screen -x you_session_id` to attach to your session again; or just `screen -r` to quick restore your session.

### Port Forwarding 
> If your server supports X11, which enables X11 forwarding, and thus make it possible to open graphics application remotely. You can just start the `firefox` or other browsers installed on your server and access Jupyter. But there are some issues:
> 1. Local MacOS system may encounter some error since there are incompatibilities between Linux.
> 2. It will take a long time for graphic transformation, usually unresponsive.

The solution is to use port forwarding, opening your browser locally and access through ssh.

#### MacOS/Linux

1. Open your terminal, and type this:
	```
	ssh -L 2222:server:22 -L 8889:server:8889 sshgate
	```
	-  You should change the `server` and `sshgate` to the real name of your server and portal
	- An example for students and staffs in University of Edinburgh is that
		```
		 ssh -L 2222:w****:22 -L 8889:w***:8889 uui@sshgate.see.ed.ac.uk
		```
	- here `w****` is your dedicated server, and `uui` is your school id
	- The command is to establish an ssh connection to SSHGate.
	- `-L` means to forward a port on your local machine. In our example, connecting to local port `2222` will be the same as connecting to port `22` on the server; which is the same to port `8889`.
2. After that, if you successfully established the connection and Jupyter is ready on your server, you can access the Jupyter on your browser by typing:
	```
	127.0.0.1:8999
	```

#### Windows

A better way in Windows is using [PuTTY](https://www.putty.org/). After installing it, you can enter the hostname of SSHGate. An example for University of Edinburgh:

![PuTTY](../../_media/PuTTY.jpeg)

Next, go to `Connection->SSH->Tunnels`, forward local port `2222` to port `22`, the same for `8889`:

![Tunnel](../../_media/tunnel.jpg)

Feel free to save the configuration, you don't want to do that again and again. Finally, click Open to open the ssh session to portal. A terminal window will appear where you can enter your username and password. Once logged in portal, just ssh to your server as usual to start the jupyter notebook.

#### Terminus

[Termius] is a ssh platform works on multiple desktop operation system, besides iOS and Android. It will sync all your configuration if you have multiple machines, so that you don't have to configure for all of them. If you feel like using graphic interface like most people, you should try this [^2].

By the way, if your are student, GitHub Studentpack provides the [free Termius access](https://termius.com/education/?utm_source=github+termius) while you are still a student.

![termius](../../_media/termius.jpg)

After your login, in the `Host` panel, you should add SSHGate and your machine accordingly. The picture shows your the example, don't forget using SSH Forwarding to quick access your server:

![termius_setup](../../_media/termius_setup.png)

You can access your server by just clicking the server host without manually entering SSHGate authentication.

Next, change your panel to `Port Forwarding`:

![termius_forwarding.jpg](../../_media/termius_forwarding.jpg)

It is very easy to use, just double click the gray `L` button, while you established VPN connection, it will turn green and everything is ok. You don't have to connect `Hosts` every time, just turn on the port forwarding.

For more information, you check on this [^3].

## Reference

1. [Remote jupyter notebooks with ssh port forwarding](https://thedatafrog.com/en/articles/remote-jupyter-notebooks/)
2. [Port Forwarding -- Terminus Documents](https://support.termius.com/hc/en-us/articles/4402386576793--Port-Forwarding#:~:text=To%20begin%2C%20open%20the%20the%20Port%20Forwarding%20screen,may%20provide%20a%20label%20for%20the%20forwarded%20port.)
3. [Port forwarding using Termius | NEK-RA](https://nek-ra.github.io/blog/termius-port-forwarding/)
