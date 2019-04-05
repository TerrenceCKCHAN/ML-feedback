import React, { Component } from 'react';
import * as THREE from 'three';


class ThreeExp extends Component{




    constructor(props) {
        super(props);

        this.state = {
            objx: null,
        };
    }

  componentDidMount(){
    const width = this.mount.clientWidth
    const height = this.mount.clientHeight
    //ADD SCENE
    this.scene = new THREE.Scene()
    //ADD CAMERA
    this.camera = new THREE.PerspectiveCamera(
      75,
      width / height,
      0.1,
      1000
    )
    this.camera.position.z = 4
    //ADD RENDERER
    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setClearColor('#000000')
    this.renderer.setSize(width, height)
    this.mount.appendChild(this.renderer.domElement)

    //ADD MODEL

    // var objectLoader = new THREE.ObjectLoader();
    // 	objectLoader.load( "teapot-claraio.json", function ( obj ) {
	//  	this.scene.add( obj );
	// } );




    var loader = new THREE.ObjectLoader();
    
    // fetch('http://localhost:3001', {
    //     mode: "no-cors"
    // })
    // .then(response => response.json())
    // .then(data => this.setState({ objx: data }))
    // .catch(err => console.error('Caught Error' + err));

    // var obj = loader.parse(this.state.objx);
    // this.scene.add(obj);
	loader.load('teapot3.json', function ( obj ) {
	 	this.scene.add( obj );
	} );


    // loader.load('gltfteapot.glb', handle_load);
    
    // function handle_load(geometry, materials) {
    //     // var geometry = new THREE.Geometry();
    //     const material = new THREE.MeshBasicMaterial({color: '#453F21'});
    //     var mesh = new THREE.Mesh(geometry, material);
    //     this.mesh.position.z = -10;
    //     mesh.scale.set(0.1,0.1,0.1);
    //     this.scene.add(mesh);
         
    // } 
    
    //ADD CUBE
    const geometry = new THREE.BoxGeometry(1, 1, 1)
    const material = new THREE.MeshBasicMaterial({ color: '#433F81'     })
    this.cube = new THREE.Mesh(geometry, material)
    // this.scene.add(this.cube)
this.start()
  }
componentWillUnmount(){
    this.stop()
    this.mount.removeChild(this.renderer.domElement)
  }
start = () => {
    if (!this.frameId) {
      this.frameId = requestAnimationFrame(this.animate)
    }
  }
stop = () => {
    cancelAnimationFrame(this.frameId)
  }
animate = () => {
   this.cube.rotation.x += 0.01
   this.cube.rotation.y += 0.01
   this.renderScene()
   this.frameId = window.requestAnimationFrame(this.animate)
 }
renderScene = () => {
  this.renderer.render(this.scene, this.camera)
}
render(){
    return(
      <div
        style={{ width: '400px', height: '400px' }}
        ref={(mount) => { this.mount = mount }}
      />
    )
  }
}
export default ThreeExp