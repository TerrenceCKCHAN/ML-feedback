import React, { Component } from 'react';
import ReactDOM from 'react-dom';

class Gl extends React.Component {

    constructor(props) {
        super(props);
        this.main = this.main.bind(this);
        this.glCanvas = React.createRef();
    }

    main() {
        // const canvas = <canvas id="glCanvas" width="640" height="480"/>;
        // Initialize the GL context
        // const gl = canvas.getContext("webgl");
      
        // Only continue if WebGL is available and working
        // if (gl === null) {
        //   alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        //   return;
        // }
      
        // Set clear color to black, fully opaque
        // gl.clearColor(0.0, 0.0, 0.0, 1.0);
        // Clear the color buffer with specified clear color
        // gl.clear(gl.COLOR_BUFFER_BIT);
      }

    render() {
        return (
            <div>
              <h1>This is Gl</h1>
              {/* <canvas ref={this.glCanvas} id="glCanvas" width="640" height="480"></canvas> */}
              {this.main()}
            </div>
        )
    }
}

export default Gl;