# Import required packages.
from asyncio.log import logger
import numpy as np
import pandas as pd 
from pathlib import Path
from dbfread import DBF
import geopandas as gpd
import copy
from .base import disagg_model

#
# The following packages that were used in the Jupyter notebook are required here:
#
# from pathlib import Path
# from dbfread import DBF
# import matplotlib.pyplot as plt
# import folium

# Symbolic constants
#
METERS_PER_MILE = 1609.34
SQUARE_METERS_PER_SQUARE_MILE = 2.59e+6
#
ONE_MILE_BUFFER = METERS_PER_MILE
HALF_MILE_BUFFER = (0.5*METERS_PER_MILE)
QUARTER_MILE_BUFFER = (0.25*METERS_PER_MILE)
#
# Enumeration constants for TDM23 transit 'MODES'
#
MODE_LOCAL_BUS = 1
MODE_EXPRESS_BUS = 2
MODE_BUS_RAPID_TRANSIT = 3
MODE_LIGHT_RAIL = 4
MODE_HEAVY_RAIL = 5
MODE_COMMUTER_RAIL = 6
MODE_FERRY = 7
MODE_SHUTTLE = 8
MODE_RTA_LOCAL_BUS = 9
MODE_REGIONAL_BUS = 10
#
# Enumeration constants for Population+Employment Density classification
#
POP_EMP_DENSITY_HIGH = 1        # pop-emp density > 10,000 per sqmi
POP_EMP_DENSITY_MEDIUM = 2      # pop-emp density > 5,000 per sqmi and < 10,000 per sqmi
POP_EMP_DENSITY_LOW = 3         # pop-emp density < 5,000 per sqmi
POP_EMP_DENSITY_NO_DATA = 0
#
# Enumeration constants for Transit Access Density classifications
#
TAD_CLASS_CBD = 1
TAD_CLASS_DENSE_URBAN = 2
TAD_CLASS_URBAN = 3
TAD_CLASS_FRINGE_URBAN = 4
TAD_CLASS_SUBURBAN = 5
TAD_CLASS_RURAL = 6


class pre_processor(disagg_model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)    
        if "init" in kwargs:
            # print (kwargs)
            if kwargs["init"]:
                self.init_logger()
        else:
            pass

        logger = self.add_logger(name=__name__)
        self.logger = logger
        logger.debug("arguments passed in %s" %kwargs)

    def __enter__(self):
        self.logger.debug("arguments passed in %s" %vars(self))
        # self.__init__(**kwargs)
        return self
        
    def __exit__(self, exc_type, exc_value, tb):
        logger = self.logger
        logger.debug("error when exit  %s,%s,%s" %(exc_type, exc_value, tb))
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    
    # Utility function: dump geodataframe to shapefile
    def dump_gdf_to_shapefile(self, gdf, shapefile_fq_path):
        gdf.to_file(shapefile_fq_path)
    # dump_gdf_to_shapefile()
    
    
    def calculate_transit_access_density_from_notebook(self):
        """ 
        NOTE: THIS IS OBSOLETE CODE
            calculate_transit_access_density_from_hotebook: 
            This is the old ('everything in one huge function') implementation of calculating transit access density.
            It is a slightly modified transcription of Margaret Atkinson's Jupyter notebook.
            It is being replaced by 'calc_transit_access_density.'
            
            For each TAZ, calcuate its "transit access density",
            and save these values in an output CSV file and shapefile.
            Tthe transit access density of a TAZ may be one of 6 possible values:
                1. CBD (central business district)
                2. Dense Urban
                3. Urban
                4. Fringe Urban
                5. Suburban
                6. Rural
        inputs: 
                1. TAZ shapefile
                2. output_folder\_networks\"Routes" shapefile
                3. output_folder|_networks\"Stops" shapefile
                4. CSV file containing socioeconomic data
                5. output_folder\_networks\CSV file listing bus stops with < 5 minute headway
                6. output_folder\_networks\CSV file listing bus stop with < 15 minute headway
        outputs:
                1. output_folder\area_types.shp shapefile
                2. output_folder\area_types.csv file
                3. database "access_density" table (TBD)
        returns:
                None
        """
        # *** QUESTION: Need to point to the appropriate sub-folder of the output folder.
        #               Do we even need to dump out the CSV if we're writing it to the DB?
        out_folder = self.args["OutputFolder"] 
        
        # Path to the TAZ 'shapefile'.
        # Note that we need to read the TAZ shapefile _itself_; reading the MA_taz_geography
        # database table isn't adequate, because it doesn't contain any spatial data.
        # SQLite doesn't support any spatial data types.
        path_taz =  self.args["geo"]
        self.logger.debug('Using TAZ shapefile: ' + path_taz)
                
        # All transit routes shapefile
        path_routes = self.args["OutputFolder"] + '\\_networks\\' + 'Routes 2022-03-17.shp'
        self.logger.debug('Using transit routes shapefile: ' + path_routes)
        
        # All stops shapefile (must have line column)
        path_stops = self.args["OutputFolder"] + '\\_networks\\' + 'Stops 2022-04-06.shp'
        self.logger.debug('Using stops shapefile: ' + path_stops)
        
        # 5-min headway and 15- min headway CSV files 
        path_f5 = self.args["OutputFolder"] + '\\_networks\\' + 'Aggr_Stops_5minheadway.csv'
        path_f15 = self.args["OutputFolder"] + '\\_networks\\' + 'Aggr_Stops_15minheadway.csv'
        self.logger.debug('Using 5-minute headway file: ' + path_f5)
        self.logger.debug('Using 15-minute headway file: ' + path_f15)
        
        #
        # Read input data
        #
        # 1. TAZ shapefile
        #
        taz = gpd.read_file(path_taz)
        self.logger.debug('Number of records in TAZ shapefile = ' + str(len(taz)))
        
        # Filter out all fields except 'id' (i.e., TAZ ID), Shape_Area, and geometry
        taz = taz[['id','geometry', 'Shape_Area']]
        # Rename 'id' to TAZ_ID
        taz = taz.rename(columns={"id":"TAZ_ID"})
        # Convert area from square meters to square miles
        taz['AREA'] = taz['Shape_Area'] / SQUARE_METERS_PER_SQUARE_MILE
        
 
        # Socio-economic data: per-TAZ 2019 total population, total employment, and total number of households
        # In the TDM19 world, this data was captured in a spreadsheet: "Statewide_2016_RTP_Updated4Proj.csv".
        #     se_2016 = r"C:\Shared drives\TMD_TSA\Model\estimation\data\tdm19_sedata\Statewide_2016_RTP_Updated4Proj.csv"
        # In the TDM23 world, the relevant data will be read from the database and assembled into a datafame.
        # 
        # Load the "hh" database table into a dataframe
        query_string = "SELECT * from hh;"
        hh_df = self.db._raw_query(qry=query_string)
  
        # Remove un-needed columns
        hh_df = hh_df.drop(columns=['block_id', 'hid', 'hh_inc', 'children', 'seniors', 'drivers'])
        
        # Renamne 'taz_id' column to 'TAZ_ID', to facilitate merge with socioeconomic data, below.
        hh_df = hh_df.rename(columns={"taz_id":"TAZ_ID"})
    
        # Get per-TAZ total population, total employment, and number of households
        # QnD way to facilitate getting #hh's per TAZ
        hh_df['num_hh'] = 1
        aggregate_se_data = hh_df.groupby(['TAZ_ID']).agg(TOT_POP = ( 'persons', 'sum' ),
                                                          TOT_HH  = ( 'num_hh', 'sum' ),
                                                          TOT_EMP = ( 'workers', 'sum' ))
        self.logger.debug('Number of records (TAZes) in aggregate SE data: ' + str(len(aggregate_se_data)) + '\n')
        
        # NOTE: The TDM23 SE data (e.g., the "hh" table) does NOT contain a record for every TAZ
        #       in the TAZ shapefile, whereas this was the case in the TDM19 world.
        #       Margaret's original code did not specify the 'kind' of join performed by the 
        #       next statement; consequently the default 'type' was used, i.e., "inner."
        #       This worked successfully in the TDM19 world. 
        #       In the TDM23 world, we must performj a "left" join in order not to loose any records
        #       in the TAZ data frame that does not have a match in the 'hh' dataframe.
        taz = pd.merge(left=taz, right=aggregate_se_data, how='left', left_on='TAZ_ID', right_on='TAZ_ID')
        self.logger.debug('Number of records in TAZ-joined-to-SE-data: ' + str(len(taz)))
               
        # 3. Read all transit routes
        # 
        # In TDM23, each scenario will have its own 'routes' and 'stops' shapefile.
        # There is no need to know the name of the scenario on which to query 
        # 'all-inclusive' 'routes' and 'stops' shapefiles.
        routes = gpd.read_file(path_routes)
        
        
        # 4. Read all stops (must have 'LINE' column)
        #
        stops = gpd.read_file(path_stops) 
        stops = stops[stops['ROUTE_ID'].isin(routes['ROUTE_ID'])] 
        
        # Transform routes, and stops layers' CRS to CTPS Standard SRS: 'EPSG:26986'" (Massachusetts State Plane, NAD 83, meters)
        stops = stops.to_crs("EPSG:26986")
        routes = routes.to_crs("EPSG:26986")
        
        # TDM23 change: join stops to routes in order to get 'MODE' of service at each stop
        all_stops_df = pd.merge(left=stops, right= routes, how="left", left_on='ROUTE_ID', right_on='ROUTE_ID')
               
        # Prune un-needed fields from all_stops_df, and rename 'geometry_x' column to 'geometry'
        all_stops_df = all_stops_df.drop(columns=['PASS_COUNT', 'MILEPOST', 'DISTANCETO', 
                                                  'FARE_ZONE', 'AVAILABLE_y', 
                                                  'TIME_NEXT', 'geometry_y'])                                                 
        all_stops_df = all_stops_df.rename(columns={'geometry_x' : 'geometry', 'AVAILABLE_x' : 'AVAILABLE'})    
        

      
        # 5. Read 5-min headway and 15-min headway CSV files
        #
        f5 = pd.read_csv(path_f5, header = 0, names = ['NODE', 'FREQ5']) #data does not come with header for either csv
        f15 = pd.read_csv(path_f15, header = 0, names = ['NODE', 'FREQ15'])        
        
        
        
        # Create individual data frames from the 'stops' dataframe for 
        # the (1) heavy rail rapid transit, (2) commuter rail, and (3) light rail rapid transit modes.
        #
        # Heavy rail rapid transit (CBD, DENSE URBAN, FRINGE URBAN)
        #    Filter all stops to just heavy rail stops (and make a copy)
        #
        hr_stops = copy.deepcopy(all_stops_df.loc[all_stops_df['MODE'].isin([MODE_HEAVY_RAIL])]) 
        # Remove un-needed fields
        hr_stops = hr_stops.loc[:,['ID', 'MODE', 'ROUTE_NAME', 'STOP_NAME', 'NEAR_NODE', 'geometry']]      
          
        # Commuter Rail (DENSE URBAN, FRINGE URBAN)
        #    Filter all stops to just commuter rail stops (and make a copy)
        cr_stops = copy.deepcopy(all_stops_df.loc[all_stops_df['MODE'].isin([MODE_COMMUTER_RAIL])]) 
        # Remove un-needed fields
        cr_stops = cr_stops.loc[:,['ID', 'MODE', 'ROUTE_NAME', 'STOP_NAME', 'NEAR_NODE', 'geometry']] 
         
        # Light Rail (FRINGE URBAN & CBD)
        #    Filter all stops to just light rail stops (and make a copy)
        lr_stops = copy.deepcopy(all_stops_df.loc[all_stops_df['MODE'].isin([MODE_LIGHT_RAIL])]) 
        # Remove un-needed fields
        lr_stops = lr_stops.loc[:,['ID', 'MODE', 'ROUTE_NAME', 'STOP_NAME', 'NEAR_NODE', 'geometry']]  
        
        # Light Rail subset: the 4 Green Line stops carrying all 4 Green Line routes
        # Change for TDM23: the name of the Park Street Green Line and Copley stations has changed from the TDM19 world.
        #
        # The TDM19 code read as follows:
        #     all4_lrStop = copy.deepcopy(lr_stops.loc[self.lr_stops_df['STOP_NAME'].isin(
        #                                         ['PARK STREET GREEN', 'BOYLSTON', 'ARLINGTON', 'COPLEY SQUARE'])]) 
        #
        all4_lrStop = copy.deepcopy(lr_stops.loc[lr_stops['STOP_NAME'].isin(
                                            ['PARK STREET', 'BOYLSTON', 'ARLINGTON', 'COPLEY'])])         
                
            
        # Remove un-needed fields
        all4_lrStop = all4_lrStop.loc[:,['ID', 'MODE', 'ROUTE_NAME', 'STOP_NAME', 'NEAR_NODE', 'geometry']]
        
        
        # Just get Nodes from Stops (DENSE URBAN, FRINGE URBAN)
        # 
        # ** QUESTION: TDM19 --> TDM23 change uses 'NEAR_NODE' rather than 'NODE' in Stops data layer
        #
        nodes = copy.deepcopy(all_stops_df.groupby('NEAR_NODE').first().reset_index()) # this makes it one record per node id
        nodes = nodes[['NEAR_NODE', 'geometry']]
        
        # Merge with the service frequency data
        #
        # Note: The following merge statements have been changed from what they were in the TDM19 world
        #     nodes = nodes.merge(f5, how = 'left', on = 'NODE')
        #     nodes = nodes.merge(f15, how = 'left', on = 'NODE')
        nodes = pd.merge(left=nodes, right=f5, how='left', left_on='NEAR_NODE', right_on='NODE')
        nodes = pd.merge(left=nodes, right=f15, how='left', left_on='NEAR_NODE', right_on='NODE')
        #
        # Cleanup artifacts of the 2 merges:
        nodes = nodes.drop(columns=['NODE_y'])
        nodes = nodes.rename(columns={'NODE_x' : 'NODE'})
        
        # Drop any record with 3 or more NULL values in the _joined_ columns - this is a shortcut. (M. Atkinson)
        nodes = nodes.dropna(how = 'any', thresh=3) 
        nodesf5 = nodes.query('FREQ5 >= 36') # this filters so headway < 5min and only bus modes 1,2,3 (because thats whats in the file)
        nodesf15 = nodes.query('FREQ15 >= 12') # this filters so headway < 15min   
        
        # Routes for SUBURBAN
        # 
        # tdm19_sub_routes = copy.deepcopy(routes.loc[routes['MODE'].isin([1,2,3,4,5,6,7,8,12,13,17,18,19,20,21,22,41,42,43])]) 
        #
        # *** QUESTION: Double check my 'translation' of the mode list from TDM19-speak into TDM23-speak is correct.
        #
        sub_routes = copy.deepcopy(routes.loc[routes['MODE'].isin([1,2,3,4,5,9])])
        
        
        # Calculate Density for TAZs
        # Density = (Population + Employment)/Area in Sq Mi
        taz['Pop_Emp_Density'] = (taz['TOT_POP']+taz['TOT_EMP'])/(taz['AREA'])
        
        # Sanity check: Are there any TAZes with a NULL 'Pop_Emp_Density'?
        no_pe_density_df = taz[taz['Pop_Emp_Density'].isnull()]
        
        print('Number of records with NULL Pop_Emp_Density: ' + str(len(no_pe_density_df)))
        
        
        # 3-way classificaiton of each TAZ's 'population + employment' density:
        #     1 --> high
        #     2 --> medium
        #     3 --> low
        #
        # This was Margaret's code:
        #
        # taz['Den_Flag'] = np.where(taz['Pop_Emp_Density'] >= 10000, 1, np.where(taz['Pop_Emp_Density'] < 5000, 3, 2))
        
        def classify_pop_emp_density(row):
            if row['Pop_Emp_Density'] >= 10000:
                retval = 1
            elif row['Pop_Emp_Density'] >= 5000:
                retval = 2
            else:
                retval = 3
            #
            return retval
        # end_def
        taz['Den_Flag'] = taz.apply(lambda row: classify_pop_emp_density(row), axis=1)
        
        # Margaret's "sanity check / visual check" - useless outside of a notebook
        # taz.loc[taz['Den_Flag']<3]
        
        ###############################################################################
        #
        # Step 1: Identify 'CBD' TAzes
        #
        # 1/2 Mile Buffer
        #
        # Convert the 0.5 mile buffer distance into meters (CRS is in meters)
        buf5 = 0.5*METERS_PER_MILE
        
        # Make a copy of hr_stops to buffer
        hr_buf = copy.deepcopy(hr_stops) 
        
        # Buffer the heavy rail rapid transit stops
        # Keep attribute data by replacing the geometry column
        hr_buf['geometry']= hr_buf.buffer(HALF_MILE_BUFFER)
        
        # Group by line and dissolve
        # NOTE: Change here for difference between TDM19 and TDM23 data schemas:
        # Rather than dissolving by 'MODE' (all heavy rail rapid transit lines have the same MODE in TDM23),
        # here, we dissolve by ROUTE_NAME.
        hr_buf_dis = hr_buf.dissolve(by='ROUTE_NAME').reset_index()
        # Keep only the fields of interest
        hr_buf_dis = hr_buf_dis[['MODE', 'ROUTE_NAME', 'geometry']]
        
        # 1/4 Mile Buffer
        #
        # Convert the 0.25 mile buffer distance into meters (CRS is in meters)
        buf25 = 0.25*METERS_PER_MILE
        
        # Make a copy of lr_stops to buffer
        lr4_buf = copy.deepcopy(all4_lrStop) 
        
        # Buffer the 'subset' light rail stops
        lr4_buf['geometry']= lr4_buf.buffer(buf25)
        
        # Group and dissolve
        # *** QUESTION: Unlike the case of heavy rail rapid transit, dissolving by MODE here should be OK, no?
        lr4_buf_dis = lr4_buf.dissolve(by='MODE').reset_index()
        # Keep only the fields of interest
        lr4_buf_dis = lr4_buf_dis[['MODE', 'ROUTE_NAME', 'geometry']].reset_index()
        
        # Get individual geo-dataframes for each of the 3 heavy rail rapid transit lines 
        #
        # In the TDM23 world, there is a single mode number for 'heavy rail rapid transit'; 
        # in the TDM19 world there there was a single 'mode' for EACH heavy rail rapid transit line. (Go figure.)
        # Here we need to create individual geo-dataframes for buffered stops for Red, Orange, and Blue lines.
        # Each such line can be identified by the ROUTE_NAME field, but note that there are distinct ROUTE_NAME values.
        # For example these are the possible 'ROUTE_NAMEs' that identify the Red line:
        #    1. 'Red Line (Alewife - Ashmont):Red'
        #    2. 'Red Line (Ashmont - Alewife):Red'
        #    3. 'Red Line (Alewife - Braintree):R'
        #    4. 'Red Line (Braintree - Alewife):R'
        # Note that those trailing "R's" (rather that 'Red') in (3) and (4) appear to be legit,
        # possibly due to some field length limitation in TransCAD.
        # The long and the short of it is that we have to be careful when filtering to select these lines.
        #
        # Get individual geodataframes for each of the 3 heavy rail rapid transit lines ... to be intersected
        # TDM19-speak for the following statements was: 
        #    red = hr_buf_dis[hr_buf_dis['MODE'] == 5]
        #    blue = hr_buf_dis[hr_buf_dis['MODE'] == 7]
        #    orange = hr_buf_dis[hr_buf_dis['MODE'] == 8]
        red = hr_buf_dis[hr_buf_dis['ROUTE_NAME'].str.startswith('Red Line')] 
        blue = hr_buf_dis[hr_buf_dis['ROUTE_NAME'].str.startswith('Blue Line')] 
        orange = hr_buf_dis[hr_buf_dis['ROUTE_NAME'].str.startswith('Orange Line')] 
        
        # Get all overlaps between route buffers
        rb = gpd.overlay(red, blue, how='intersection')
        ob = gpd.overlay(orange, blue, how='intersection')
        ro = gpd.overlay(red, orange, how= 'intersection')
        
        # Put overlaps into one geodataframe with overlaps not overlapping (union)
        rbob = gpd.overlay(rb, ob, how='union')
        rbobro = gpd.overlay(rbob, ro, how='union',keep_geom_type=False)
        
        # Add in the 4 stops of the Green Line light rail carrying all 4 Green Line 'branches' ('B', 'C', 'D', and 'E')
        rbobro = rbobro.append(lr4_buf)
        
        # Turn into a single multi part polygon, i.e., a geodataframe with one row
        #
        # The next statement isolates Red, Orange, Blue, and the 4 'main' stops of the Green Line.
        # This is accomplished by an UNABASHED TRICK: setting the 'MODE' for these records to 
        # the synthetic value 'red_orange_blue_green_4', and dissolving on the 'MODE' field.
        # (In the TDM19 world, the synthetic value used here for the 'MODE' field was '578+g4'.)
        rbobro['MODE'] = 'red_orange_blue_green_4' 
        rbobro = rbobro.dissolve(by='MODE').reset_index()
        # Get only fields we need
        rbobro = rbobro[['MODE', 'geometry']]
        
        # Select the TAZes that intersect with 'red_orange_blue_green_4'
        taz_hr = gpd.overlay(taz, rbobro, how='intersection')
        
        print("Number of TAZes in 'rapid transit buffer': " + str(len(taz_hr)))
        
        # Caculate the total area of these TAZes
        #
        taz_hr['area_of_intersection'] = taz_hr.area
        taz['taz_area'] = taz.area
        
        # Join the 'TAZes intersecting with the "R,B,O,G4"' dataframe (i.e., taz_hr') to the main TAZ dataframe,
        # and calculate the percentage of each TAZ that intersects with the "R,B,O,G4" dataframe.
        taz_int = taz.merge(taz_hr, how='left', on='TAZ_ID')
        
        
        # Indicate 'CBD' TAZes:
        # If more than 50% of a TAZ intersects with "R,B,O,G4", flag it as a 'CBD' TAZ. 
        #
        taz_int['perc_hr'] = taz_int['area_of_intersection']/taz_int['taz_area']       
        taz_int['CBD_Flag'] = np.where(((taz_int.perc_hr >0.5)), 1, 0)
        
        print("Number ot TAZes with 'CBD_Flag==1': = " + str(len(taz_int[taz_int['CBD_Flag'] == 1])))
        
        # Turn 'taz_int' back into a geodataframe:
        #
        # The 'merge' above merged two geodataframes, each with a 'geometry' column.
        # The result of the merge was a vanilla dataframe with a 'geometry_x' and a 'geometry_y' column.
        # The dataframe has to have a 'geometry' field in order to be a geodataframe and be usable
        # in geographic calculations. Rename the 'geometry_x' column to 'geometry'.
        #
        taz_int = taz_int.rename(columns={'geometry_x':'geometry', 'TOT_EMP_x' : 'TOT_EMP', 'TOT_POP_x': 'TOT_POP',
                                          'Pop_Emp_Density_x':'Pop_Emp_Density', 'Den_Flag_x': 'Den_Flag'})
        taz_int = gpd.GeoDataFrame(taz_int)
        taz_int = taz_int[['TAZ_ID', 'TOT_EMP', 'TOT_POP', 'geometry', 'Pop_Emp_Density', 'Den_Flag', 'CBD_Flag']]
        
        # Make sure that taz_int doesn't have duplicates because of light rail and heavy rail being different buffer polygons.
        # First sort so that when deleting duplicates, we delete the ones that didn't pass CBD muster
        taz_int = taz_int.sort_values(by = ['CBD_Flag'], ascending = False)
        # Delete duplicates
        taz_int = taz_int.drop_duplicates('TAZ_ID')
            
        ###############################################################################
        #
        # Steps 2 and 3: Identify 'Dense Urban' and 'Urban' TAZes.
        #
        # Buffering commuter rail, heavy rail rapid transit, light rail rapid transit
        # and buses with < 5 minute headway for 'Dense Urban'.
        #
        # Convert the 0.5 mile buffer distance into meters (CRS is in meters)
        buf5 = 0.5*METERS_PER_MILE
        
        # Make copies of everything to buffer
        cr_buf = copy.deepcopy(cr_stops) 
        hr_buf = copy.deepcopy(hr_stops) 
        lr_buf = copy.deepcopy(lr_stops)
        nodesf5_buf = copy.deepcopy(nodesf5)
        
        # Buffer everything
        cr_buf['geometry']= cr_buf.buffer(buf5)
        hr_buf['geometry']= hr_buf.buffer(buf5)
        lr_buf['geometry']= lr_buf.buffer(buf5)
        nodesf5_buf['geometry'] = nodesf5_buf.buffer(buf5)
        
        # Dissolve hr and cr and bus stops with headway < 5 min
        hr_buf_dis_du = hr_buf.dissolve().reset_index()
        cr_buf_dis = cr_buf.dissolve().reset_index()
        lr_buf_dis = lr_buf.dissolve().reset_index()
        nodesf5_buf_dis = nodesf5_buf.dissolve().reset_index()
        
        # *** QUESTION: We're talking about *two* fields here, no? ('DU_Flag' and 'URB_Flag')
        # *** Per Margaret: Question for Marty
        #
        # Calculate Flag Field
        #
        taz_du = copy.deepcopy(taz_int) #join basic taz with taz_int
        
        # These lists will contain the TAZ IDs of the TAZes that intersect with
        # commuter rail, heavy rail rapid transit, light riail, and buses with 5-minute headways.
        cr_list = []
        hr_list = []
        f5_list = []
        lr_list = []
        for index, row in gpd.overlay(hr_buf_dis_du, taz_du, how = 'intersection').iterrows():
            hr_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(lr_buf_dis, taz_du, how = 'intersection').iterrows():
            lr_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(cr_buf_dis, taz_du, how = 'intersection').iterrows():
            cr_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(nodesf5_buf_dis, taz_du, how = 'intersection').iterrows():
            f5_list.append(row['TAZ_ID'])
        #
        taz_du['DU_Flag'] = np.where(
                                    ((taz_du['TAZ_ID'].isin(hr_list)) & (taz_du['Den_Flag'] == 1)) | 
                                    ((taz_du['TAZ_ID'].isin(lr_list)) & (taz_du['Den_Flag'] == 1)), 1,0)
                                    
        taz_du['URB_Flag'] = np.where(((taz_du['TAZ_ID'].isin(cr_list)) & (taz_du['Den_Flag'] == 1)) | ((taz_du['TAZ_ID'].isin(f5_list)) & (taz_du['Den_Flag'] == 1)), 1,0)
                               
        ###############################################################################
        #
        # Step 4 Identify 'Fringe Urban' TAZes.
        #
        #
        # Buffering everything for Fringe Urban
        #
        # Convert the 0.5 and 1 mile buffer distances into meters (CRS is in meters)
        buf5 = 0.5*METERS_PER_MILE
        buf1 = 1*METERS_PER_MILE # obvious but for clarity
        
        # Make copies of everything to buffer
        cr_buf = copy.deepcopy(cr_stops) 
        hr_buf1 = copy.deepcopy(hr_stops)   # hr_buf - 1 mile
        nodesf15_buf = copy.deepcopy(nodesf15)
        lr_buf = copy.deepcopy(lr_stops)
        
        # Buffer everything
        cr_buf['geometry']= cr_buf.buffer(buf5)
        hr_buf1['geometry']= hr_buf1.buffer(buf1) #buffer is now 1 mi
        nodesf15_buf['geometry']= nodesf15_buf.buffer(buf5)
        lr_buf['geometry']= lr_buf.buffer(buf5)
        
        # dissolve hr and cr
        hr_buf_dis_fu = hr_buf1.dissolve().reset_index()
        cr_buf_dis = cr_buf.dissolve().reset_index()
        nodesf15_buf_dis = nodesf15_buf.dissolve().reset_index()
        lr_buf_dis = lr_buf.dissolve().reset_index()
        
        #
        # Calculate Flag Field
        taz_fu = copy.deepcopy(taz_du) #join basic taz with taz_int
        
        # These lists will contain the TAZ IDs of the TAZes that intersect with
        # commuter rail, heavy rail rapid transit, light riail, and buses with 15-minute headways.
        cr_list = []
        hr_list = [] #1 mile buffer now
        f15_list = []
        lr_list = []
        # For each row in the intersection output - take all the TAZs (will only be ones that intersect) and put them in a list
        for index, row in gpd.overlay(hr_buf_dis_fu, taz_fu, how = 'intersection').iterrows():
            hr_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(cr_buf_dis, taz_fu, how = 'intersection').iterrows():
            cr_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(nodesf15_buf_dis, taz_fu, how = 'intersection').iterrows():
            f15_list.append(row['TAZ_ID'])
        #
        for index, row in gpd.overlay(lr_buf_dis, taz_fu, how = 'intersection').iterrows():
            lr_list.append(row['TAZ_ID'])
        #
        # Set 'FU_Flag' if fany of the following conditions are met
        taz_fu['FU_Flag'] = np.where((taz_fu['TAZ_ID'].isin(hr_list)) | 
                                     (taz_fu['TAZ_ID'].isin(lr_list)) |
                                     ((taz_fu['TAZ_ID'].isin(cr_list)) & (taz_fu['Den_Flag'] == 2)) | 
                                     ((taz_fu['TAZ_ID'].isin(f15_list)) & (taz_fu['Den_Flag'] == 2)), 1,0)
        taz_fu.query('FU_Flag == 1') #should be 482 (no CBD or DU), 1188 with CBD and DU
        
        ###############################################################################
        #
        # Step 5 Identify 'Suburban' TAZes.
        # 
        # Do the buffers first
        # Convert miles to meters for CRS
        buf5 = 0.5*METERS_PER_MILE
        #make copy to copy buffer geo into
        sub_buf = copy.deepcopy(sub_routes)
        #do the buffers
        sub_buf['geometry']= sub_buf.buffer(buf5)
        #dissolve the buffers
        sub_buf_dis = sub_buf.dissolve().reset_index()
        
        #
        # Calculate 'SUB_Flag'  field
        taz_su = copy.deepcopy(taz_fu)
        #make list of all taz that intersect with sub_buf_dis
        sub_list = []
        for index, row in gpd.overlay(sub_buf_dis, taz_su, how = 'intersection').iterrows():
            sub_list.append(row['TAZ_ID'])
        #
        
        # Flag everything that intersects with one of the buffers (included in list)
        taz_su['SUB_Flag'] = np.where(taz_su['TAZ_ID'].isin(sub_list), 1, 0)
        taz_su.query("SUB_Flag == 1").plot()
        
        ###############################################################################
        #
        # Step 6 Identify 'Rural' TAZes.
        #
        # Rural TAZes are those that are neither CBD, Dense Urban, Urban, 
        # Fringe Urban nor Suburban.
        #
        taz_ru = copy.deepcopy(taz_su)
        taz_ru['R_Flag'] = np.where((taz_ru['CBD_Flag'] == 0) & (taz_ru['DU_Flag'] == 0) & 
                                    (taz_ru['URB_Flag'] == 0) & (taz_ru['FU_Flag'] == 0) & (taz_ru['SUB_Flag'] == 0), 1, 0)

        # Create 'LU_6Level_Type' column for 6-level land use category.
        # This was previously called 'LU_6Level_Type', but has been renamed 'TA6'.
        #
        # 6-level type codes: CBD = 1, U_RT = 2, U_CRB = 3, FU = 4, SUB = 5, Rural = 6 
        taz_ru['TA6'] = np.where(taz_ru['CBD_Flag'] == 1, 1, 
                                 np.where((taz_ru['CBD_Flag'] != 1) & (taz_ru['DU_Flag'] == 1), 2,
                                 np.where((taz_ru['CBD_Flag'] != 1) & (taz_ru['DU_Flag'] != 1) & (taz_ru['URB_Flag'] == 1), 3,
                                 np.where((taz_ru['CBD_Flag'] != 1) & (taz_ru['DU_Flag'] != 1) & (taz_ru['URB_Flag'] != 1) & (taz_ru['FU_Flag'] == 1), 4,
                                 np.where((taz_ru['CBD_Flag'] != 1) & (taz_ru['DU_Flag'] != 1) & (taz_ru['URB_Flag'] != 1) & (taz_ru['FU_Flag'] != 1) & (taz_ru['SUB_Flag'] == 1), 5, 6 
                                 )))))
         
        # The above statement produces a floating-point value, convert it to an integer value.
        taz_ru = taz_ru.astype({"TA6": int})

        output_shapefile_fp = out_folder + r"\area_type.shp"
        taz_ru.to_file(output_shapefile_fp)
        
        output_tabular_df = taz_ru[["TAZ_ID","TA6"]]
        output_csv_fp = out_folder + r"\area_type.csv"
        output_tabular_df = output_tabular_df.sort_values(by=["TAZ_ID"])
        output_tabular_df.to_csv(output_csv_fp, index=False)
        #
        print("TO DO: Export output_tabular_df to database 'access_density' table.")
        #
        
        # Collect and print some summary information:
        n_cbd = len(output_tabular_df[taz_ru['TA6'] == 1])
        n_dense_urb = len(taz_ru[taz_ru['TA6'] == 2])
        n_urb = len(taz_ru[taz_ru['TA6'] == 3])
        n_fringe_urb = len(taz_ru[taz_ru['TA6'] == 4])
        n_sub = len(taz_ru[taz_ru['TA6'] == 5])
        n_rural = len(taz_ru[taz_ru['TA6'] == 6])
        total_tazes = n_cbd + n_dense_urb + n_urb + n_fringe_urb + n_sub + n_rural
        
        data = { 'TA6_classification' : ['1', '2', '3', '4', '5', '6', 'total'],
                 'count'              : [n_cbd, n_dense_urb, n_urb, n_fringe_urb, n_sub, n_rural, total_tazes] }
        summary_df = pd.DataFrame(data)
        self.logger.debug("Contents of transit access density summary DF:\n")
        self.logger.debug(summary_df.head(10))
        
        print("Contents of transit access density summary DF:\n")
        print(summary_df.head(10))       
        
        summary_csv_fn = self.args['OutputFolder']  + '\\_logs\\' + 'access_density_summary.csv'
        summary_df.to_csv(summary_csv_fn, index=False)  

        # DEVELOPMENT/DEBUG - compare calculated results from 'estimated' TA6 values
        
        # Load estimated TA6 classification into a dataframe
        ta6_estimated_fn = out_folder + r"\area_type_ta6.csv"
        ta6_estimated_df = pd.read_csv(ta6_estimated_fn)
        ta6_estimated_df = ta6_estimated_df.rename(columns={"TA6":"TA6_est"})
        # Join to compare
        comparison_df = pd.merge(left=output_tabular_df, right=ta6_estimated_df, how='outer', left_on='TAZ_ID', right_on='TAZ_ID')
        comparison_df['mismatch'] = comparison_df.apply(lambda row: 1 if row['TA6'] != row['TA6_est'] else 0, axis=1)
        comparison_df.sort_values(by=['TAZ_ID'])
        comparison_csv_fn = self.args['OutputFolder']  + '\\access_density_COMPARISON.csv'
        comparison_df.to_csv(comparison_csv_fn, index=False)        
        
        # Isloate records with differences:
        difference_df = comparison_df[comparison_df['mismatch'] == 1] 
        # Remove the 'mismatch' column, as it's not needed in the 'differences only' CSV file
        difference_df = difference_df.drop(columns=['mismatch'])
        difference_csv_fn = self.args['OutputFolder']  + '\\access_density_DIFFERENCES.csv'
        difference_df.to_csv(difference_csv_fn, index=False)
        
    # end_def calcuate_transit_access_density()
    
# end_class pre_processor()

if __name__ == "__main__":
    pp = pre_processor()
    # pp.calculate_transit_access_density()
