"""
Performance Report Viewer
View and analyze performance reports and metrics
"""
import json
import sys
import os
from pathlib import Path
from config import config

def view_latest_report():
    """View the latest performance report"""
    log_dir = Path(config.LOG_PATH)
    if not log_dir.exists():
        print("No logs directory found")
        return
    
    # Find latest report
    reports = list(log_dir.glob("performance_report_*.txt"))
    if not reports:
        print("No performance reports found")
        return
    
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    
    print(f"\n{'='*80}")
    print(f"Viewing: {latest.name}")
    print(f"{'='*80}\n")
    
    with open(latest, 'r') as f:
        print(f.read())

def view_final_report():
    """View the final performance report"""
    report_path = Path(config.LOG_PATH) / "final_performance_report.txt"
    
    if not report_path.exists():
        print("No final performance report found")
        return
    
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE REPORT")
    print(f"{'='*80}\n")
    
    with open(report_path, 'r') as f:
        print(f.read())

def view_metrics_json():
    """View metrics JSON file"""
    json_path = Path(config.LOG_PATH) / "final_metrics.json"
    
    if not json_path.exists():
        print("No metrics JSON found")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80 + "\n")
    
    stats = data.get("stats", {})
    
    print(f"Runtime: {stats.get('runtime_seconds', 0):.2f} seconds")
    print(f"Frames: {stats.get('frame_count', 0):,}")
    print(f"Episodes: {stats.get('episode_count', 0):,}")
    print(f"Training Steps: {stats.get('total_training_steps', 0):,}")
    
    if 'fps' in stats:
        fps = stats['fps']
        print(f"\nFPS: {fps.get('current', 0):.2f} (avg: {fps.get('average', 0):.2f})")
    
    if 'latencies_ms' in stats:
        lat = stats['latencies_ms']
        print(f"\nLatencies (ms):")
        print(f"  Capture: {lat.get('capture', {}).get('avg', 0):.2f}")
        print(f"  Inference: {lat.get('inference', {}).get('avg', 0):.2f}")
        print(f"  Injection: {lat.get('injection', {}).get('avg', 0):.2f}")
        print(f"  Qwen: {lat.get('qwen', {}).get('avg', 0):.2f}")
    
    if 'episodes' in stats:
        ep = stats['episodes']
        print(f"\nEpisode Performance:")
        print(f"  Average Reward: {ep.get('average_reward', 0):.2f}")
        print(f"  Best Reward: {ep.get('best_reward', 0):.2f}")
        print(f"  Total Episodes: {ep.get('total_episodes', 0)}")
    
    print(f"\nErrors: {stats.get('errors', {}).get('total', 0)}")
    print(f"Warnings: {stats.get('warnings', {}).get('total', 0)}")

def list_all_reports():
    """List all available reports"""
    log_dir = Path(config.LOG_PATH)
    if not log_dir.exists():
        print("No logs directory found")
        return
    
    reports = sorted(log_dir.glob("performance_report_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not reports:
        print("No performance reports found")
        return
    
    print(f"\nFound {len(reports)} performance reports:\n")
    for i, report in enumerate(reports[:10], 1):  # Show last 10
        size = report.stat().st_size
        mtime = report.stat().st_mtime
        from datetime import datetime
        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {report.name} ({size:,} bytes) - {date_str}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "latest":
            view_latest_report()
        elif command == "final":
            view_final_report()
        elif command == "json":
            view_metrics_json()
        elif command == "list":
            list_all_reports()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python view_performance.py [latest|final|json|list]")
    else:
        # Default: show latest
        view_latest_report()

if __name__ == "__main__":
    main()

