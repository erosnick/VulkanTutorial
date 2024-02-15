#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 texcoord;

layout(binding = 1) uniform sampler2D textureSampler;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(textureSampler, texcoord);

    // outColor = vec4(texcoord, 0.0, 1.0);
}