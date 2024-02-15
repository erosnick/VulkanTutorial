#version 450

layout(Location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
} UBO;

layout(location = 0) out vec3 fragColor;

void main()
{
    gl_Position = UBO.projection * UBO.view * UBO.model * vec4(inPosition, 0.0, 1.0);

    fragColor = inColor;
}