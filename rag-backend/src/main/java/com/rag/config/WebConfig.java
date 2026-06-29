package com.rag.config; // 配置类所在包。

import org.springframework.context.annotation.Configuration; // 声明这是 Spring 配置类。
import org.springframework.web.servlet.config.annotation.CorsRegistry; // 配置跨域规则。
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry; // 配置静态资源映射。
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry; // 配置简单页面转发。
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer; // Spring MVC 扩展接口。

import java.nio.file.Path; // 用来拼接前端目录的本地文件路径。

@Configuration // 让 Spring Boot 启动时加载这个配置类。
public class WebConfig implements WebMvcConfigurer { // 实现 WebMvcConfigurer 后，可以自定义 MVC 行为。

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 把前端目录映射为静态资源
        Path frontendPath = Path.of("../frontend").toAbsolutePath().normalize();

        // /assets/** 下的文件直接从 frontend/ 目录提供（CSS、JS 等）
        registry.addResourceHandler("/assets/**")
                .addResourceLocations("file:" + frontendPath + "/");

        // /index.html 也从 frontend/ 目录提供
        registry.addResourceHandler("/index.html")
                .addResourceLocations("file:" + frontendPath + "/");
    }

    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        // 访问根路径 / 时，转发到 /index.html
        registry.addViewController("/").setViewName("forward:/index.html");
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        // 允许前端访问 /api/** 接口。
        // 这个项目里前端和 Java 都在 8080，一般不会跨域；保留它是为了开发调试方便。
        registry.addMapping("/api/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "POST", "DELETE", "OPTIONS")
                .allowedHeaders("*");
    }
}
