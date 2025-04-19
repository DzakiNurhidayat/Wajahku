import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import streamlit as st
from app.pages.capture import main as capture_page

def main():
    capture_page()

if __name__ == "__main__":
    main()
