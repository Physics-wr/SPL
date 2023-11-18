### Self-Promptable Segmentation for Generalizable Person Re-identification

Implementation of "Self-Promptable Segmentation for Generalizable Person Re-identification"

how to run:

1. change FASTREID_DATASETS in train.sh to the path of datasets.

2. modify configs/sam_reid.yml so that the model can load SAM weights

3. train the model using:

   ```bash
   sh train.sh	
   ```

   