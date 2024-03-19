# import numpy as np
# import cv2
# from pathlib import Path
# # import cvlib as cv
# # from cvlib.object_detection import draw_bbox
# import matplotlib.pyplot as plt
# from keras.models import load_model

# BASE_DIR = Path(__file__).resolve().parent
# # model = load_model(f"{BASE_DIR}/model.hdf5", compile=False)

import os, sys
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from rasterio.enums import Resampling

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestRegressor

def predict(input):
    data_path = input
    data_path
    
    def plot_band_array(band_array, image_extent, title, cmap_title, colormap, colormap_limits):
    # Use masked array to handle invalid values (NaN)
        masked_array = np.ma.masked_invalid(band_array)

    plt.imshow(masked_array, extent=image_extent)
    cbar = plt.colorbar()
    plt.set_cmap(colormap)
    plt.clim(colormap_limits)
    cbar.set_label(cmap_title, rotation=270, labelpad=20)
    plt.title(title)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    rotatexlabels = plt.setp(ax.get_xticklabels(), rotation=90)

    def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,epsg):

        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

    def raster2array(geotif_file):
        metadata = {}

        dataset = gdal.Open(geotif_file)
        metadata['array_rows'] = dataset.RasterYSize
        metadata['array_cols'] = dataset.RasterXSize
        metadata['bands'] = dataset.RasterCount
        metadata['driver'] = dataset.GetDriver().LongName
        metadata['projection'] = dataset.GetProjection()
        metadata['geotransform'] = dataset.GetGeoTransform()

        mapinfo = dataset.GetGeoTransform()
        metadata['pixelWidth'] = mapinfo[1]
        metadata['pixelHeight'] = mapinfo[5]

        metadata['ext_dict'] = {}
        metadata['ext_dict']['xMin'] = mapinfo[0]
        metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
        metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]
        metadata['ext_dict']['yMax'] = mapinfo[3]

        metadata['extent'] = (metadata['ext_dict']['xMin'],metadata['ext_dict']['xMax'],
                            metadata['ext_dict']['yMin'],metadata['ext_dict']['yMax'])

    if metadata['bands'] == 1:
        raster = dataset.GetRasterBand(1)
        metadata['noDataValue'] = raster.GetNoDataValue()
        metadata['scaleFactor'] = raster.GetScale()

        # Check if scaleFactor is None
        if metadata['scaleFactor'] is not None:
            # band statistics
            metadata['bandstats'] = {} # make a nested dictionary to store band stats in the same
            stats = raster.GetStatistics(True, True)
            metadata['bandstats']['min'] = round(stats[0], 2)
            metadata['bandstats']['max'] = round(stats[1], 2)
            metadata['bandstats']['mean'] = round(stats[2], 2)
            metadata['bandstats']['stdev'] = round(stats[3], 2)

            array = dataset.GetRasterBand(1).ReadAsArray(0,0,
                                                         metadata['array_cols'],
                                                         metadata['array_rows']).astype(float)
            array[array == int(metadata['noDataValue'])] = np.nan
            array = array / metadata['scaleFactor']
            return array, metadata
        else:
            print("Scale factor is None. Division not performed.")
            return None, metadata
    else:
        print('More than one band ... function only set up for single-band data')

    def crown_geometric_volume_pct(tree_data,min_tree_height,pct):
        p = np.percentile(tree_data, pct)
        tree_data_pct = [v if v < p else p for v in tree_data]
        crown_geometric_volume_pct = np.sum(tree_data_pct - min_tree_height)
        return crown_geometric_volume_pct, p

    def get_predictors(tree,chm_array, labels):
        indexes_of_tree = np.asarray(np.where(labels==tree.label)).T
        tree_crown_heights = chm_array[indexes_of_tree[:,0],indexes_of_tree[:,1]]

    full_crown = np.sum(tree_crown_heights - np.min(tree_crown_heights))

    crown50, p50 = crown_geometric_volume_pct(tree_crown_heights,tree.min_intensity,50)
    crown60, p60 = crown_geometric_volume_pct(tree_crown_heights,tree.min_intensity,60)
    crown70, p70 = crown_geometric_volume_pct(tree_crown_heights,tree.min_intensity,70)

    return [tree.label,
            float(tree.area),  # use float instead of np.float
            tree.major_axis_length,
            tree.max_intensity,
            tree.min_intensity,
            p50, p60, p70,
            full_crown,
            crown50, crown60, crown70]

    chm_file = input
    chm_file

    #Get info from chm file for outputting results
    chm_name = os.path.basename(chm_file)

    chm_array, chm_array_metadata = raster2array(chm_file)


    with rasterio.open(chm_file) as src:
        chm_array = src.read(1)  # Assuming it's a single-band image

    # Replace NaN values with 0 and convert to float
    chm_array = np.nan_to_num(chm_array).astype(float)

    # Plot the CHM figure
    plot_band_array(chm_array, chm_array_metadata['extent'],
                'Canopy Height Model',
                'Canopy Height (m)',
                'Greens', [0, 9])

    # Save the plot as a PNG file
    plt.savefig(os.path.join(data_path, chm_name.replace('.tif', '_processed.png')),
                dpi=300, orientation='landscape',
                bbox_inches='tight', pad_inches=0.1)

    # Show the plot
    plt.show()

    #Smooth the CHM using a gaussian filter to remove spurious points
    chm_array_smooth = ndi.gaussian_filter(chm_array,2,mode='constant',cval=0,truncate=2.0)
    chm_array_smooth[chm_array==0] = 0

    #Save the smoothed CHM
    array2raster(os.path.join(data_path,'chm_filter.tif'),
             (chm_array_metadata['ext_dict']['xMin'],chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(chm_array_smooth,dtype=float),32611)
    
    #Calculate local maximum points in the smoothed CHM
    local_maxi = peak_local_max(chm_array_smooth,indices=False, footprint=np.ones((5, 5)))

    local_maxi

    #local_maxi.astype(int)
    local_maxi_numeric = local_maxi.astype(int)

    print(local_maxi_numeric)


    plt.figure(2)
    plot_band_array(local_maxi_numeric, chm_array_metadata['extent'],
                'Maximum',
                'Maxi',
                'viridis',  # Change the colormap to viridis
                [0, 1])  # Adjust color limits if needed

    plt.savefig(data_path + chm_name[0:-4] + '_Maximums.png',
            dpi=300, orientation='landscape',
            bbox_inches='tight', pad_inches=0.1)


    # Save the numerical array to a raster
    array2raster(data_path + 'maximum.tif',
             (chm_array_metadata['ext_dict']['xMin'], chm_array_metadata['ext_dict']['yMax']),
             1, -1, np.array(local_maxi_numeric, dtype=np.float32), 32611)
    
    #Identify all the maximum points
    markers = ndi.label(local_maxi)[0]


    #Create a CHM mask so the segmentation will only occur on the trees
    chm_mask = chm_array_smooth
    chm_mask[chm_array_smooth != 0] = 1

    #WATERSHED_SEGMENTATION

    #Perform watershed segmentation
    labels = watershed(chm_array_smooth, markers, mask=chm_mask)
    labels_for_plot = labels.copy()
    labels_for_plot = np.array(labels_for_plot, dtype=float)
    labels_for_plot[labels_for_plot==0] = np.nan
    max_labels = np.max(labels)

    #Plot the segments
    plot_band_array(labels_for_plot,chm_array_metadata['extent'],
                'Crown Segmentation','Tree Crown Number',
                'Spectral',[0, max_labels])

    plt.savefig(data_path+chm_name[0:-4]+'_Segmentation.png',
            dpi=300,orientation='landscape',
            bbox_inches='tight',pad_inches=0.1)

    array2raster(data_path+'labels.tif',
             (chm_array_metadata['ext_dict']['xMin'],
              chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(labels,dtype=float),32611)  # use float instead of np.float
    

    #Get the properties of each segment
    tree_properties = regionprops(labels,chm_array)

    predictors_chm = np.array([get_predictors(tree, chm_array, labels) for tree in tree_properties])
    X = predictors_chm[:,1:]
    tree_ids = predictors_chm[:,0]

    np.shape(predictors_chm)


    #TRAINING DATA

    folder_name = "biomass_data"
    training_data_file = os.path.join('/content/', folder_name, 'SJER_Biomass_Training.csv')

    training_data = np.genfromtxt(training_data_file, delimiter=',')

    biomass = training_data[:, 0]

    biomass_predictors = training_data[:, 1:12]


    #RANDOM FOREST CLASSIFIER

    #Define parameters for the Random Forest Regressor
    max_depth = 30

    #Define regressor settings
    regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)

    #Fit the biomass to regressor variables
    regr_rf.fit(biomass_predictors,biomass)


    #Apply the model to the predictors
    estimated_biomass = regr_rf.predict(X)


    #Set an out raster with the same size as the labels
    biomass_map =  np.array((labels),dtype=float)
    #Assign the appropriate biomass to the labels
    biomass_map[biomass_map==0] = np.nan
    for tree_id, biomass_of_tree_id in zip(tree_ids, estimated_biomass):
        biomass_map[biomass_map == tree_id] = biomass_of_tree_id

    

    #BIOMASS CALCULATION
        
    #Get biomass stats for plotting
    mean_biomass = np.mean(estimated_biomass)
    std_biomass = np.std(estimated_biomass)
    min_biomass = np.min(estimated_biomass)
    sum_biomass = np.sum(estimated_biomass)

    # print('Sum of biomass is ',sum_biomass,' kg')

    # Plot the biomass!
    plt.figure(5)
    plot_band_array(biomass_map,chm_array_metadata['extent'],
                'Biomass (kg)','Biomass (kg)',
                'winter',
                [min_biomass+std_biomass, mean_biomass+std_biomass*3])

    # Save the biomass figure; use the same name as the original file, but replace CHM with Biomass
    plt.savefig(os.path.join(data_path,chm_name.replace('CHM.tif','Biomass.png')),
            dpi=1000,orientation='landscape',
            bbox_inches='tight',
            pad_inches=0.1)


    # Use the array2raster function to create a geotiff file of the Biomass
    array2raster(os.path.join(data_path,chm_name.replace('CHM.tif','Biomass.tif')),
             (chm_array_metadata['ext_dict']['xMin'],chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(biomass_map,dtype=float),32611)
    
    # Calculate estimated carbon storage
    estimated_carbon_storage = np.sum(estimated_biomass) * 0.5  # Assuming 50% carbon content

    ('Estimated Carbon Storage is {:.2f} kg'.format(estimated_carbon_storage))



    #BIOMASS DENSITY & CARBON DENSITY




    tif_path = '/content/biomass_data/NEON_D17_SJER_DP3_256000_4106000_CHM.tif'

    with rasterio.open(tif_path) as src:
        data = src.read(1)

    # Get the affine transformation parameters
    transform = src.transform

    # Calculate area in square meters
    pixel_area = abs(transform.a * transform.e)
    total_area_meters = (data > 0).sum() * pixel_area

    # Convert square meters to square kilometers
    total_area_km2 = total_area_meters / 1e6

    # print(f"Total Forest Area in square meters: {total_area_meters:.2f} square meters")

    # print(f'Total Forest Area: {total_area_km2:.2f} square kilometers')
    # print(f" ")


    # Calculate biomass density and carbon density
    biomass_density = sum_biomass / total_area_meters  # in kg/m²
    carbon_density = estimated_carbon_storage / total_area_meters  # in kg/m²

    # print(f"Biomass Density: {biomass_density} kg/m²")
    # print(f"Carbon Density: {carbon_density} kg/m²")




    return sum_biomass, estimated_carbon_storage, biomass_density, carbon_density, input_img
    # return None
