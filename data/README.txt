Due to the large size of the dataset we have not uploaded it in the repository.
However, in this folder we provide samples of data for each dataset and respecting
the original folder structure. It can be found: 

- MSG_ML: This folder contains the "raw" satellite and radar images in NETCDF file format.
This images have been obtained from the original radar and satellite archives using the 
MeteoSwiss-developed software package Pyrad.
From the original POH radar data a mask has been created by thresholding the data with
a 90% probability threshold. The masked area have been extended by masking as well the
surrounding pixels to account for the difference in satellite and radar spatial resolution.
The resultant image has been cropped to the area of interest for this study and saved in a
NETCDF file.
The original satellite data was contained in a NETCDF file with a single file containing all
channels per time step. From this original data the channel differences and the texture have
been computed. The HRV channel has been normalized by the sun zenith angle. The resultant
image has been cropped to the area of interest for this study. For convenience each feature
has been saved in a different NETCDF file.

- rf_data: This folder contains the data pre-processed to be ingested into the RF models family
in a .npz file.
Each file contains two numpy arrays. One containing the features matrix and one containing the
targets matrix. A file contains data extracted over an entire month. The feature matrix has 6 columns
corresponding to normalized HRV, normalized HRV texture, IR 10.8 um, IR 10.8 um texture,
window channel and window channel texture. The target matrix has a single column with values
0 (no POH computed), 1 (POH below 90%) and 2 (POH equal or above 90%)

- rf_data expanded: As above but with 2 new features: IR 1.6 um and its texture 

 
- dl_data: This folder contains the pre-processed data for the unet model in an .npz file.
Each file contains one time step of data. The file contains two numpy arrays. A feature
matrix of the form (nx, ny, nchannels) and a target matrix of the form nx, ny, nclasses.
The channels are the normalized HRV, IR 10.8 um and window channel. They are min-max normalized
so that they range from 0 to 1. The classes are 0 (no hail) and 1 (hail).

- dl_data_expanded: As above but adding the feature IR 1.6 um.     
