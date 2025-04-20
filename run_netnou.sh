#!/bin/bash

# NetNou - AI Student Attendance & Engagement Analysis
# Helper script to run common commands

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Make sure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print usage information
print_usage() {
    echo -e "${BLUE}NetNou Helper Script${NC}"
    echo "Usage: ./run_netnou.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  analyze          - Run real-time face and engagement analysis"
    echo "  analyze:fast     - Run analysis optimized for speed"
    echo "  analyze:accurate - Run analysis optimized for accuracy"
    echo "  train            - Train the engagement neural network"
    echo "  webapp           - Run the web application"
    echo "  setup            - Install dependencies"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_netnou.sh analyze"
    echo "  ./run_netnou.sh analyze:fast"
    echo "  ./run_netnou.sh train"
}

# Ensure the script is executable
if [[ ! -x "$0" ]]; then
    chmod +x "$0"
    echo -e "${GREEN}Made script executable${NC}"
fi

# Check if argument is provided
if [[ $# -eq 0 ]]; then
    print_usage
    exit 1
fi

# Process the command
case "$1" in
    "analyze")
        echo -e "${GREEN}Starting face and engagement analysis...${NC}"
        python NetNou/demographic_analysis/live_demographics.py
        ;;
    "analyze:fast")
        echo -e "${GREEN}Starting optimized analysis for speed...${NC}"
        python NetNou/demographic_analysis/live_demographics.py --detector ssd --analyze_every 3
        ;;
    "analyze:accurate")
        echo -e "${GREEN}Starting optimized analysis for accuracy...${NC}"
        python NetNou/demographic_analysis/live_demographics.py --detector retinaface --analyze_every 1
        ;;
    "train")
        echo -e "${GREEN}Training engagement neural network...${NC}"
        python NetNou/scratch_nn/train_engagement_nn.py
        ;;
    "webapp")
        echo -e "${GREEN}Starting web application...${NC}"
        if [ -d "NetNou-WebApp" ]; then
            cd NetNou-WebApp
            python run.py
        else
            echo -e "${YELLOW}WebApp directory not found. Have you cloned the repository?${NC}"
            exit 1
        fi
        ;;
    "setup")
        echo -e "${GREEN}Installing dependencies...${NC}"
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python -m venv venv
        
        # Activate virtual environment based on OS
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source venv/Scripts/activate
        else
            source venv/bin/activate
        fi
        
        echo -e "${YELLOW}Installing required packages...${NC}"
        pip install -r requirements.txt
        echo -e "${GREEN}Setup complete! You can now run other commands.${NC}"
        ;;
    "help")
        print_usage
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac

exit 0 