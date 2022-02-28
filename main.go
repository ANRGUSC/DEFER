package main

import (
	"fmt"

	"k8s.io/client-go/kubernetes"
)

func main() {
	kubernetes.NewForConfig()
	fmt.Print("Hi")
}
