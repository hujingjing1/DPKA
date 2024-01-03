# DPKA Sparse-View CT Reconstruction

## Installation

To set up your environment for the DPKA model, you will need to install PyTorch, the ASTRA toolbox, and ODL. Follow the steps below:

1. Install the ASTRA toolbox using conda:
conda install -c astra-toolbox astra-toolbox.
3. Install the development version of ODL (Operator Discretization Library) using pip:
pip install https://github.com/odlgroup/odl/archive/master.zip 
4. To train the DPKA model with your dataset, execute the following command:
python train.py
5. This script will start the training process using the parameters specified within the file.
## Testing

For testing the model and evaluating its performance, use:
python test.py

## Acknowledgements

Our work on the DPKA model has been inspired by and builds upon prior research in the field. We have referenced and adapted methodologies from the following significant works:

- Wu W, Hu D, Niu C, et al. "DRONE: Dual-domain residual-based optimization network for sparse-view CT reconstruction." IEEE Transactions on Medical Imaging, 2021, 40(11): 3002-3014.
- Zhang Y, Li D, Shi X, et al. "Kbnet: Kernel basis network for image restoration." arXiv preprint arXiv:2303.02881, 2023.
- Mei Y, Fan Y, Zhou Y. "Image super-resolution with non-local sparse attention." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 3517-3526.





