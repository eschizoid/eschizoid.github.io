It's 2026. Your Java object mappings are still strings. They shouldn't be.

For ten years MapStruct has had you name every field in a string literal the compiler never checks. Rename a getter, it compiles green, then breaks in production. I got tired of it and built the alternative.

Telescope: your field mappings are typed method references. Rename the field, the IDE follows it, javac catches the miss. Bidirectional from one definition, deep nested updates, validation as a first-class effect, and codegen that benchmarks at parity with MapStruct.

It's on Maven Central. Write your next mapper with it, and watch the difference the first time you rename a field.

👉 https://mariano-gonzalez.com/posts/post-7/

#java #softwareengineering #mapstruct

<!-- Reach tip: LinkedIn tends to suppress posts with outbound links. For more views, put the
     https://mariano-gonzalez.com/posts/post-7/ link in the FIRST COMMENT instead of the body. -->
