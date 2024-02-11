// 1.CreateWindow.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/matrix.hpp>

#include <iostream>

class HelloTriangleApplication
{
public:
    void run();

private:
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

private:
    GLFWwindow* window;
};

void HelloTriangleApplication::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void HelloTriangleApplication::initWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
}

void HelloTriangleApplication::initVulkan()
{
	uint32_t extensionCount = 0;

	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::cout << extensionCount << " extensions supported" << std::endl;
}

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
	glfwDestroyWindow(window);

	glfwTerminate();
}

void HelloTriangleApplication::cleanup()
{

}

int main()
{
    try
    {
        HelloTriangleApplication app;

        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}