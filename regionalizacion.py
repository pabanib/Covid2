# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:20:32 2022

@author: Pablo
"""

import pandas as pd
import geopandas as gpd 
import lq
import elegir_modelo as em
import sklearn

class regionalizacion(lq.evaluaciones_lq):
    
    def __init__(self, panel_df,grupos, variables, poblacion):
        super().__init__(panel_df)
        self.variables = variables
        self.poblacion = poblacion
                
    def add_pipeline(self, pipeline):
        
        assert isinstance( pipeline,sklearn.pipeline.Pipeline ), 'Debe ingresar un pipeline de sklearn'
        self.pipeline = pipeline
        
    def ajustar_datos(self):

        pass
       
    
    
    
#%%    
    
reg = regionalizacion('f','v','p')

reg.add_pipeline(pipe)
