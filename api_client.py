#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description='Glaze Recipe Solver API Client')
    parser.add_argument('--server', type=str, default='http://localhost:5000', help='API server URL')
    parser.add_argument('--action', type=str, required=True, 
                        choices=['solve', 'umf_to_weights', 'weights_to_umf', 'health'],
                        help='API action to perform')
    parser.add_argument('--umf', type=str, help='UMF composition as JSON string')
    parser.add_argument('--weights', type=str, help='Weight composition as JSON string')
    parser.add_argument('--solutions', type=int, default=3, help='Number of solutions to find (default: 3)')
    parser.add_argument('--min-materials', action='store_true', help='Prefer solutions with fewer materials')
    parser.add_argument('--error-tolerance', type=float, default=0.01, 
                        help='Acceptable error increase for solutions with fewer materials (default: 0.01)')
    
    args = parser.parse_args()
    
    base_url = args.server.rstrip('/')
    
    try:
        if args.action == 'health':
            response = requests.get(f"{base_url}/api/health")
            print_response(response)
            
        elif args.action == 'solve':
            if not args.umf:
                print("error: umf parameter is required for solve action")
                return
            
            try:
                umf_data = json.loads(args.umf)
            except json.JSONDecodeError:
                print("error: invalid JSON format for umf parameter")
                return
            
            payload = {
                "umf": umf_data,
                "max_solutions": args.solutions,
                "min_materials": args.min_materials,
                "error_tolerance": args.error_tolerance
            }
            
            response = requests.post(f"{base_url}/api/solve", json=payload)
            print_response(response)
            
        elif args.action == 'umf_to_weights':
            if not args.umf:
                print("error: umf parameter is required for umf_to_weights action")
                return
            
            try:
                umf_data = json.loads(args.umf)
            except json.JSONDecodeError:
                print("error: invalid JSON format for umf parameter")
                return
            
            payload = {"umf": umf_data}
            response = requests.post(f"{base_url}/api/umf_to_weights", json=payload)
            print_response(response)
            
        elif args.action == 'weights_to_umf':
            if not args.weights:
                print("error: weights parameter is required for weights_to_umf action")
                return
            
            try:
                weights_data = json.loads(args.weights)
            except json.JSONDecodeError:
                print("error: invalid JSON format for weights parameter")
                return
            
            payload = {"weights": weights_data}
            response = requests.post(f"{base_url}/api/weights_to_umf", json=payload)
            print_response(response)
            
    except requests.exceptions.RequestException as e:
        print(f"error: connection error: {e}")

def print_response(response):
    """Print formatted API response"""
    try:
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"error: server returned status code {response.status_code}")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
    except Exception as e:
        print(f"error: failed to parse response: {e}")

if __name__ == "__main__":
    main() 