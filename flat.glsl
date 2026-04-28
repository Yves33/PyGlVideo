#version 440
uniform sampler2D   u_tex0;

#if defined MGLVERTEXSHADER || defined VERTEX_SHADER
#line __LINE__
layout (location=0) in vec2 a_position;
layout (location=1) in vec2 a_uv;

// vertex shader output data
out vData{
    vec2        a_uv;
}vsout;

void main()
{
    vsout.a_uv=a_uv;   // copy texture coordinates
    gl_Position = vec4(a_position,0.0,1.0);
}
#endif  //MGLVERTEXSHADER

#if defined MGLFRAGMENTSHADER || defined FRAGMENT_SHADER
#line __LINE__
// fragment shader input
in vData{
    vec2        a_uv;   // texture coordinates
    }fsin;

out vec4 out_FragColor;
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

void mainImage_flat(out vec4 fragColor,in vec2 fragCoord) {
	//fragColor = vec4(texture2D(u_tex0, fragCoord).rgb,1.0);
    fragColor = vec4(texture2D(u_tex0, fragCoord).rgb,1.0);//+vec4(fragCoord.x,fragCoord.y,1.0,1.0);
}
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
void main ()
{
    mainImage_flat(out_FragColor,fsin.a_uv);
}
#endif
