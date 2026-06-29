package com.rag; // 当前 Java 类所在包名，Spring Boot 会从这个包往下扫描组件。

import org.springframework.boot.SpringApplication; // Spring Boot 启动器，用来启动内嵌 Web 服务。
import org.springframework.boot.autoconfigure.SpringBootApplication; // 标记这是 Spring Boot 主启动类。

@SpringBootApplication // 等价于 @Configuration + @EnableAutoConfiguration + @ComponentScan。
public class RagApplication {
    // Java 程序入口。运行这个 main 方法后，Spring Boot 会启动 8080 端口服务。
    public static void main(String[] args) {
        // 启动 Spring 容器，自动扫描 Controller、Service、Config 等组件。
        SpringApplication.run(RagApplication.class, args);
    }
}
