/*
 * Macro template to process multiple images in a folder
 */

#@ File (label = "Input directory", style = "directory") input
#@ String (label = "File suffix", value = ".tif") suffix
#@ String (label = "File prefix", value = "EdgesOfMicrofluidics_") prefix

output = input
processFolder(input);
print("Finished!");

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			if(startsWith(list[i], prefix))
				processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	print("Processing: " + input + File.separator + file);
	open(input + File.separator + file);
	id = getImageID();
	processImage(id, output);
}

function processImage(id, output) {
	selectImage(id);
	title = getTitle();
	newTitle = replace(title, prefix, "HoughLabels_");
	run("Hough Circle Transform","minRadius=145, maxRadius=160, inc=1, minCircles=1, maxCircles=4, threshold=0.6, resolution=952, ratio=1.0, bandwidth=10, local_radius=10,  reduce show_centroids");
	//wait(50000);
	waitForUser("Click OK when the Hough plugin has finished.");
	selectWindow("Centroid map");
	centroidID = getImageID();
	rename(newTitle);
	print("Saving to: " + output + File.separator + newTitle);
	run("Save", "save=[" + output + File.separator + newTitle + "]");
	close("*");
	run("Collect Garbage");
}
