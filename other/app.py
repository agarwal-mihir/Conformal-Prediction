import streamlit as st

def custom_slider():
    st.title('Custom Slider in Streamlit')

    # Embed the custom HTML for the slider
    st.markdown("""
        <div class="slider-container">
            <div class="slider" id="mySlider">
                <div class="slider-thumb" id="mySliderThumb"></div>
            </div>
            <div id="slider-value">Value of α : 0.4</div>
        </div>
    """, unsafe_allow_html=True)

    # Load and run the JavaScript code
    with open('slider.js', 'r') as f:
        js_code = f.read()
        st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)

if __name__ == "__main__":
    custom_slider()
