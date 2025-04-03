package main

import "fmt"

// Define a Person struct
type Person struct {
    Name string
    Age  int
}

// Function to greet a person
func greet(person Person) string {
    return fmt.Sprintf("Hello, %s! You are %d years old.", person.Name, person.Age)
}

func main() {
    // Creating a Person struct instance
    person := Person{
        Name: "Bob",
        Age:  25,
    }

    // Calling the greet function
    fmt.Println(greet(person))
}
