#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File extension", value=".nd2") suffix
#@ String(label="Drift correction reference channel:",choices={"1	", "2	", "3	", "4	"},value="1	",style="radioButtonHorizontal") reference_channel
#@ Boolean (label = "Multi time scale computatation (slow!)", value = false) multi_time_scale
#@ Boolean (label = "Sub pixel drift correction (for slow drifts)", value = false) sub_pixel
#@ Boolean (label = "Edge enhancement for drift correction", value = true) edge_enhance
#@ String(label="Cell segmentation channel:",choices={"1	", "2	", "3	", "4	"},value="2	",style="radioButtonHorizontal") segment_cells_channel
#@ String(label="Microfluidic segmentation channel:",choices={"1	", "2	", "3	", "4	"},value="3	",style="radioButtonHorizontal") segment_microfluidics_channel

//setBatchMode(true);
close("*");
print("\\Clear");
processFolder(input);
print("Finished!");
//setBatchMode(false);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	print("Reading image: " + input + File.separator + file);
	open(input + File.separator + file);
	wait(4000);
	//waitForUser("Has the image been read in yet?");
	selectWindow(file);
	id = getImageID();
	processImage(id);
}

//// function to scan folder to find files with correct suffix
//function processFile(filename) {
//	print("Processing: " + filename);
//	for (i = 1; i < 1000; i++) {
//		print("Reading in image series " + d2s(i, 0) + " with BioFormats...");
//		arg_series = "series_" + toString(i);
//		run("Bio-Formats Importer",
//			"open=[" + filename + "] " +
//			"color_mode=Default " +
//			"rois_import=[ROI manager] " +
//			"view=Hyperstack " +
//			"stack_order=XYCZT " +
//			"use_virtual_stack " +
//			arg_series);
//		id = getImageID();
//		processImage(id);
//		run("Collect Garbage");
//	}
}

function processImage(id) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	selectImage(id);
	idDriftCorrected = correctDrift(id);
	segmentCells(idDriftCorrected);
	idEdges = findMicrofluidicEdges(idDriftCorrected);
	//idHoughLabels = detectMicrofluidicCircles(idEdges);
	// = (idHoughLabels);
	close("*");
	run("Collect Garbage");
}

function correctDrift(id) {
	print("Correcting drift in images");
	selectImage(id);
	title = getTitle();
	correctDriftArgs = createCorrectDriftArgs(reference_channel, multi_time_scale, sub_pixel, edge_enhance);
	run("Correct 3D drift", correctDriftArgs);
	wait(10000);
	selectWindow("registered time points");
	new_id = getImageID();
	new_title = "DriftCorrected_" + replace(title, suffix, "");
	rename(new_title);
	print("Saving: " + output + File.separator + new_title + ".tif");
	save(output + File.separator + new_title + ".tif");
	return new_id;
}

// function that creates string with arguments for Correct3DDrift
function createCorrectDriftArgs(reference_channel, multi_time_scale, sub_pixel, edge_enhance) {
	ref_channel = stripWhitespace(reference_channel);
	function_call_args = "channel=" + ref_channel;
	if (multi_time_scale == true) {
		function_call_args += " multi_time_scale";
	} // no elif or else
	if (multi_time_scale == true) {
		sub_pixel += " sub_pixel";
	} // no elif or else
	if (edge_enhance == true) {
		function_call_args += " edge_enhance";
	} // no elif or else
	function_call_args += " only=0 lowest=1 highest=1";
	return function_call_args;
}

function segmentCells(id) {
	print("Segmenting cells");
	selectImage(id);
	title = getTitle();
	new_title = "SegmentedCells_" + replace(title, suffix, "");
	cell_channel = stripWhitespace(segment_cells_channel);
	run("Duplicate...", "title=" + new_title + " duplicate channels=" + cell_channel);
	new_id = getImageID();
	run("Gaussian Blur...", "sigma=1 stack");
	run("Convert to Mask", "method=Triangle background=Dark black");
	run("Watershed", "stack");
	print("Saving: " + output + File.separator + new_title + ".tif");
	save(output + File.separator + new_title + ".tif");
	close(new_id);
}


function stripWhitespace(str) {
	str = replace(str, "\\t", ""); // strip tabs
	str = replace(str, " ", ""); // strip spaces
	return str;
}

function findMicrofluidicEdges(id) {
	print("Finding microfluidic edges in image");
	selectImage(id);
	title = getTitle();
	new_title = "EdgeOfMicrofluidics_" + replace(title, suffix, "");
	microfluidic_mask_channel = stripWhitespace(segment_microfluidics_channel);
	run("Duplicate...", "duplicate channels="+microfluidic_mask_channel+" frames=1");
	new_id = getImageID();
	run("Enhance Contrast", "saturated=0.35");
	run("Gaussian Blur...", "sigma=2");
	run("Minimum...", "radius=3");
	run("Subtract Background...", "rolling=100 light");
	setOption("BlackBackground", true);
	run("Convert to Mask");
	new_title = "EdgesOfMicrofluidics_" + replace(title, ".tif", "");
	print("Saving: " + output + File.separator + new_title + ".tif");
	save(output + File.separator + new_title + ".tif");
	return new_id;
}

function detectMicrofluidicCircles(id) {
	print("Detecting circles of microfluidic wells");
	selectImage(id);
	title = getTitle();
	run("Duplicate...", "title=Edges_for_mask");
	new_id = getImageID();
	run("Hough Circle Transform",
		"minRadius=" + min_seqrch_radius + ", " +
		"maxRadius=" + max_seqrch_radius + ", " +
		"inc=1, " +
		"minCircles=" + min_n_to_find + ", " +
		"maxCircles=" + max_n_to_find + ", " +
		"threshold=" + confidence_threshold + ", " +
		"resolution=100, ratio=1.1, bandwidth=10, local_radius=10,  " +
		"reduce show_centroids");
	//	"reduce show_centroids show_scores");
	close(new_id);
	// Optional saving of hough score map
	//selectWindow("Score map");
	//new_title = "HoughScores_" + replace(title, suffix, "");
	//print("Saving: " + output + File.separator + new_title + ".tif");
	//save(output + File.separator + new_title + ".tif");
	//close();
	selectWindow("Centroid map");
	run("8-bit");
	id_hough = getImageID();
	new_title = "HoughLabels_" + replace(title, suffix, "");
	print("Saving: " + output + File.separator + new_title + ".tif");
	save(output + File.separator + new_title + ".tif");
	return id_hough;
}
