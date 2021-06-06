# Upsampling-Bilateral-Filter

Parts of the program:
- The main program implementing the ICP and TrICP is the main.cpp file. There is a CMakeLists.txt file for compilation.
  - It takes, as the first argument, the data pointcloud
  - The second argument is the model pointcloud
  - The third argument is the minimum overlap if the two pointclouds (0.0-1.0]
  - The fourth argument is the maximum number of iterations
  - The fifth argument is the output filename without extension, that is used to save the results.
  - The sixth argument is the standard deviation of the noise added to the pointclouds
  - The seventh argument is the file containing the rotation and translation of the the data file. (created by the program below)
  - An example execution can be: "./icp ../pointclouds/a_5.xyz ../pointclouds/b.xyz 0.6 1000 fountain_5_noisy 0.05 ../pointclouds/a_5_rot_trans.txt"

- There is another program in the pointclouds directory called transform_pointcloud.cpp.
  - it takes as the first argument a .xyz file and it randomly rotates and translates the points.
  - the second argument is the output filename without the extension.
  - There is a CMakeLists.txt file for compilation.
  - An example execution can be: "./transform_pointcloud a.xyz a_1" -> generates a_1.xyz file and a_1_rot_trans.txt, which contains the rotation and translation matrices used to modify the data.

- The starting pointclouds where a.xyz (which is rotated and translated by transform_pointcloud.cpp) and b.xyz.

- The results can be found in the results directory
  - It contains the resulting pointclouds after ICP execution.
  - The execution of ICP results in 4 pointclouds:
    - {output_filename}_data_ICP.xyz
    - {output_filename}_model_ICP.xyz
    - {output_filename}_data_TrICP.xyz
    - {output_filename}_model_TrICP.xyz
  - The results.txt file contains the rotation error, translation error and execution times for each execution.
