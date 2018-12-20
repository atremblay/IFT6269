# IFT6269 - Fall 2018

Probabilistic Graphical Models at the University of Montreal
http://www.iro.umontreal.ca/~slacoste/teaching/ift6269/A18/


## Session project!
Implementing [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf)

Students in alphabetical order:

- Marc-Antoine Bélanger (20114235)
- Jean-Philippe Gagnon (1016340)
- Defeng Liu (Polytechnique - 1886962)
- Alexis Tremblay (31109)


To run the code:

	You will need to download the required datasets:
		camvid: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
		nyuv2: http://dl.caffe.berkeleyvision.org/nyud.tar.gz
		make3d: http://make3d.cs.cornell.edu/data.html

	And organize the data as per the following tree where you will specify the '--data_folder' argument:
		-data\
		|  -camvid\
		|  |   -test\
		|  |   -testannot\
		|  |   -train\
		|  |   -trainannot\	
		|  |   -val\
		|  |   -valannot\
		|  |   -test.txt
		|  |   -train.txt
		|  |   -val.txt
		|  -make3d\
		|  |   -train\
		|  |   -trainDepth\
		|  |   -val\
		|  |   -valDepth\
		|  -nyuv2\
		|  |   -data\
		|  |   |   -depth\
		|  |   |   -images\
		|  |   |   -...
		|  |   -segmentation\
		|  |   -...

	Initial Training (Command line examples):

	python main.py --dataset camvid --opt rmsprop --loss hc_loss --cuda --epochs 350

	python main.py --dataset camvid --opt rmsprop --loss nll_loss --cuda --epochs 350

	python main.py --dataset nyuv2 --task classification --opt rmsprop --loss nll_loss --cuda --epochs 350

        python main.py --dataset nyuv2 --task regression --opt rmsprop --loss mse_loss --cuda --epochs 350

	
	To resume a training (Command line examples):

	python resume.py /where/the/data/was/saved/folder

	

