#version 450

layout(Location = 0) in vec3 inPosition;
layout(Location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;
layout(location = 3) in vec3 inColor;

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
} UBO;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 texcoord;

void main()
{
    gl_Position = UBO.projection * UBO.view * UBO.model * vec4(inPosition, 1.0);

    fragColor = inColor;
    texcoord = inTexcoord;
}