# Semi-Automated-High-Throughput-Truthing-of-Pathologist-Annotation-Assistant

This is the implementation code of the paper: **[Single-Image-Object-Counting-and-Localizing-using-Active-Learning](https://www.cs.huji.ac.il/w~inbarhub/projects/count_WACV/paper.pdf)*

Requirements:
tensorflow 1.14.0

### Abstract
Artificial intelligence algorithms that process whole slide images (WSIs) and determine the percentage of sTILs in a tumor cell will automatically will reduce a pathologist's burden of searching and evaluating cells and features on tumor slides. A validation dataset established by pathologist annotations is a necessary step to create this algorithm.

The goal of this project is to determine if a weakly trained Convolutional Neural Network (CNN) annotation system could act as an annotation assistant to increase data collection accuracy and efficiency by semi-determining the number of sTILs in tumor cells. 

Methods and steps in my project include altering a Simple Image Object Counting and Localizing using an Active Learning algorithm to count the number of sTILs within a tumor cell region. This pre-made algorithm takes in human feedback to re-learn and generate a count number closer to the actual count number. The “human providing the feedback” action will be done by pathologists during the annotation process in the real world. To avoid human interference in my project, I replaced this human-to-computer feedback with ground-truth-data-to-computer feedback. My system will make corrections to its predictions according to the ground truth data (GTD).

The studies and test runs support how after each algorithm learning iteration, my system computes a number closer to the ground truth number of sTILs within the region of interest. Data also concludes how my system can be efficient for Pathologists to use when annotating sTILs because the system’s prediction accuracy increases after each validation correction from the GTD.illumination and occlusions

#### Inconsistencies and room for error:
Inconsistency in the visual representation of sTILs in the cel (ROI) image can cause the annotation system to count cells that look similar to sTILs that are actually not sTILs. The ground truth data used to assess and validate the annotation system contains the estimated center XY coordinates of some of the sTILs in an ROI. This can cause uncertainties because the ground truth data is calculated by the means of the XY coordinates of the outline of the sTILs  and are not the actual center. Therefore the center of image of an sTIL for the system to learn can shift and is not completely accurate. Lastly, the ground truth data to test whether the system counted the right number of sTILs can cause uncertainties. The ground truth data does not contain count all of the sTILs in the ROI, only some. This can cause the system to count less and produce a number that is rounded down than the actual number. 


### Running the code

To run the code, use 'count_repetitive_objects.py' script with two arguments: an image name and the participant name.
For example:
* python3 count_repetitive_objects.py Cell1 Phoebe
* python3 count_repetitive_objects.py Cell34 Jonathan
* python3 count_repetitive_objects.py Cell16 Michael

The names of the images (e.g., Cell1, Cell16...) can be found in the nucls_data folder

