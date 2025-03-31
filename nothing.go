// A _line filter_ is a common type of program that reads
// input on stdin, processes it, and then prints some
// derived result to stdout. `grep` and `sed` are common
// line filters.

// Here's an example line filter in Go that writes a
// capitalized version of all input text. You can use this
// pattern to write your own Go line filters.
package main
q
import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {

	// Wrapping the unbuffered `os.Stdin` with a buffered
	// scanner gives us a convenient `Scan` method that
	// advances the scanner to the next token; which is
	// the next line in the default scanner.
	scanner := bufio.NewScanner(os.Stdin)

	for scanner.Scan() {
		// `Text` returns the current token, here the next line,
		// from the input.
		ucl := strings.ToUpper(scanner.Text())

		// Write out the uppercased line.
		fmt.Println(ucl)
	}

	// Check for errors during `Scan`. End of file is
	// expected and not reported by `Scan` as an error.
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}aa XXMqWIwKP8 aaFav7lI37 OvbK7hOVR9 HkUrNxrFmo 4xlXGKSYOg uXaW6GAcFp DOSawTI3sa 80A68iOQgI 23X7wtazhI WZQsxP1jNm OQKbuJO71E ht98lR19Tj FFN9I0XZyS maHAzxYQyK tGCrMYGO0O SnqNSLNlc9 aRUsVKnENU
