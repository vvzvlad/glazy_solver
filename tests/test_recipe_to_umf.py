#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import unittest
import sys
import os

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe_to_umf import analyze_recipe

class TestUMFCalculation(unittest.TestCase):
    
    def test_recipe_analysis_with_expected_values(self):
        """Tests if UMF calculation for the test recipe matches expected values"""
        
        # Test recipe
        test_recipe = {
            "Нефелин-сиенит VR13": 30,
            "Кварцевая мука Кварцверке W12": 20,
            "Волластонит МИВОЛЛ": 20,
            "Улексит (Химпэк)": 15,
            "Каолин КЖФ-1": 15
        }
        
        # Expected UMF values from the image
        expected_umf = {
            "SiO2": 3.144,
            "Al2O3": 0.378,
            "B2O3": 0.265,
            "Na2O": 0.143,
            "K2O": 0.086,
            "CaO": 0.717,
            "MgO": 0.048,
            "SrO": 0.005,
            "Fe2O3": 0.002,
            "TiO2": 0.003
        }
        
        # Run recipe analysis
        analysis = analyze_recipe(test_recipe)
        umf = analysis['umf']
        
        # Check each oxide with 1% tolerance
        for oxide, expected_value in expected_umf.items():
            calculated_value = umf.get(oxide, 0)
            # Calculate difference percentage
            diff_percent = abs((calculated_value - expected_value) / expected_value * 100) if expected_value != 0 else 0
            
            print(f"{oxide}: expected = {expected_value}, actual = {calculated_value}, difference = {diff_percent:.2f}%")
            
            # Check that difference doesn't exceed 1%
            self.assertLessEqual(diff_percent, 1.0, 
                                f"Value for {oxide} differs by more than 1%: expected {expected_value}, got {calculated_value}")
    
if __name__ == "__main__":
    unittest.main() 