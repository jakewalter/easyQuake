#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:44:12 2023

@author: jwalter
"""

catdf = pd.read_csv('https://jakewalter.mynetgear.com/data/pr_eqt.csv')
catdf = catdf[catdf['num_arrivals']>6]

catdf = catdf[catdf['vertical_error']<20000]
catdf = catdf.reset_index(drop=True)


