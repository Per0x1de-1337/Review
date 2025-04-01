// A simple TypeScript example

// Interface to define a shape of a Person object
interface Person {
    name: string;
    age: number;
}

// Function to greet a person
function greet(person: Person): string {
    return `Hello, ${person.name}! You are ${person.age} years old.`;
}

// Creating an object of type Person
const person: Person = {
    name: "Alice",
    age: 30
};

// Calling the greet function
console.log(greet(person));
