import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from model import create_LeNet5
from tensorflow.keras.models import Model
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(layout="wide")
st.title("𝚜𝚎𝚛𝚘-5 Digit Classifier with Layer Visualization")

@st.cache_resource
def load_model():
    model = create_LeNet5((28, 28, 1))
    model.load_weights("lenet5_weights.h5")  # Replace with your weights
    layer_names = ["C1", "S2", "C3", "S4", "C5", "Output"]
    intermediate_model = Model(inputs=model.input, outputs=[model.get_layer(name).output for name in layer_names])
    return intermediate_model

model = load_model()

st.sidebar.header("Draw a digit (0-9)")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # Invert colors
    img = img / 255.0
    input_img = img.reshape(1, 28, 28, 1)

    if st.button("Predict and Visualize"):
        layer_outputs = model.predict(input_img)
        layer_names = ["C1", "S2", "C3", "S4", "C5"]

        for name, out in zip(layer_names, layer_outputs[:-1]):
            st.subheader(f"Layer: {name}")
            filters = out.shape[-1]
            fig, axes = plt.subplots(1, min(filters, 6), figsize=(12, 4))
            for i in range(min(filters, 6)):
                axes[i].imshow(out[0, :, :, i], cmap='gray')
                axes[i].axis('off')
            st.pyplot(fig)

        # Prediction Distribution
        probs = layer_outputs[-1][0]
        st.subheader("Prediction Distribution")
        fig = go.Figure(data=[
            go.Bar(x=list(range(10)), y=probs, text=[f"{p:.2f}" for p in probs], textposition='outside')
        ])
        fig.update_layout(
            xaxis_title="Digit",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(tickmode='linear')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Predicted Digit: {np.argmax(probs)}")