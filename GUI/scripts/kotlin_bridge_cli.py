#!/usr/bin/env python3
"""
CLI wrapper for the unified optimizer to support Kotlin bridge integration.
This script provides the command-line interface that the Kotlin bridge expects.
"""

import sys
import json
import argparse
import time
import math
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.pipeline.unified_optimizer import UnifiedOptimizer  # noqa: E402
from campro.logging import get_logger  # noqa: E402

# Set up logging
logger = get_logger(__name__)


def main():
    """Main CLI entry point for Kotlin bridge integration."""
    parser = argparse.ArgumentParser(description='Unified Optimization Pipeline CLI for Kotlin Bridge')
    parser.add_argument('--input', required=True, help='Input parameters JSON file')
    parser.add_argument('--output', required=True, help='Output results JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Read input parameters
        with open(args.input, 'r') as f:
            parameters = json.load(f)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        optimizer = UnifiedOptimizer(output_dir=output_dir)
        
        # Run pipeline
        start_time = time.time()
        result = optimizer.run_pipeline(parameters)
        execution_time = time.time() - start_time
        
        # Add execution time to result
        result['execution_time'] = execution_time
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_to_list(elem) for elem in obj]
            if isinstance(obj, float):
                # Sanitize non-JSON numeric tokens
                if math.isinf(obj) or math.isnan(obj):
                    return None
                return obj
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            return obj
        
        serializable_result = convert_numpy_to_list(result)

        # Adapt result to GUI-friendly schema expected by tests
        gui_out = {}
        try:
            status = serializable_result.get('status', 'failed')
            success = status == 'success'
            gui_out['success'] = success
            gui_out['executionTime'] = round(execution_time, 3)

            # Motion law summary
            motion_law = serializable_result.get('motion_law', {})
            gui_out['motionLaw'] = {
                'nodeCount': len(motion_law.get('grid', [])) if isinstance(motion_law.get('grid', []), list) else 0,
                'discretizationType': 'LGL',
                'positionRange': [
                    (min(motion_law.get('displacement', [])) if motion_law.get('displacement') else 0),
                    (max(motion_law.get('displacement', [])) if motion_law.get('displacement') else 0)
                ],
                'velocityRange': [
                    (min(motion_law.get('velocity', [])) if motion_law.get('velocity') else 0),
                    (max(motion_law.get('velocity', [])) if motion_law.get('velocity') else 0)
                ],
                'accelerationRange': [
                    (min(motion_law.get('acceleration', [])) if motion_law.get('acceleration') else 0),
                    (max(motion_law.get('acceleration', [])) if motion_law.get('acceleration') else 0)
                ],
            }

            # Gear profiles summary
            opt = serializable_result.get('optimal_profiles', {})
            r_sun = opt.get('r_sun') or []
            r_planet = opt.get('r_planet') or []
            r_ring = opt.get('r_ring_inner') or []
            gui_out['gearProfiles'] = {
                'sunRadiusRange': [min(r_sun) if r_sun else 0, max(r_sun) if r_sun else 0],
                'planetRadiusRange': [min(r_planet) if r_planet else 0, max(r_planet) if r_planet else 0],
                'ringRadiusRange': [min(r_ring) if r_ring else 0, max(r_ring) if r_ring else 0],
                'forceTransferEfficiency': (opt.get('force_transfer_efficiency', []) or [0])[-1] if isinstance(opt.get('force_transfer_efficiency', []), list) else opt.get('force_transfer_efficiency', 0),
                'maxContactStress': opt.get('max_contact_stress', 0),
                'minGearClearance': (min(opt.get('gear_clearance', [])) if opt.get('gear_clearance') else 0),
            }

            # Solver summary (Phase 2 if available)
            gui_out['iterations'] = opt.get('iterations', 0)
            gui_out['objectiveValue'] = opt.get('objective_value', 0)
            gui_out['constraintViolation'] = opt.get('constraint_violation', 0)
            gui_out['solverStatus'] = opt.get('solver_status', status)

            # In case of failure, include error
            if not success:
                gui_out['error'] = serializable_result.get('error', 'Unknown error')
        except Exception as _e:
            # Fallback to raw result if adaptation fails
            gui_out = {
                'success': False,
                'executionTime': round(execution_time, 3),
                'error': 'Result adaptation failed'
            }

        # Write results
        with open(args.output, 'w') as f:
            json.dump(gui_out, f, indent=2)
        
        logger.info(f"Optimization completed successfully in {execution_time:.2f} seconds")
        return 0
        
    except ValueError as e:
        # Change: Handle preflight failures with structured diagnostics
        if "Initial guess violates variable bounds" in str(e):
            payload = {
                "status": "PREFAIL",
                "message": str(e),
                "hint": "Initial guess violated bounds; see x0_transfer logs",
                "execution_time": 0.0,
                "stage": "preflight_check"
            }
            try:
                with open(args.output, 'w') as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass
            logger.error(f"Preflight failed: {e}")
            return 2  # distinct from solver fail (1)
        else:
            # Other ValueError - treat as general failure
            error_result = {
                "status": "failed",
                "error": str(e),
                "execution_time": 0.0,
                "stage": "pipeline_execution"
            }
            try:
                with open(args.output, 'w') as f:
                    json.dump(error_result, f, indent=2)
            except Exception:
                pass
            logger.error(f"Optimization failed: {e}")
            return 1
    except Exception as e:
        # Create error result
        error_result = {
            "status": "failed",
            "error": str(e),
            "execution_time": 0.0,
            "stage": "pipeline_execution"
        }
        
        try:
            with open(args.output, 'w') as f:
                json.dump(error_result, f, indent=2)
        except Exception:
            pass
        
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
