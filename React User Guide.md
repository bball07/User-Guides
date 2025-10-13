# React User Guide

## What is React?

**React** created by **Facebook (now Meta)**  is a **JavaScript library** that creates **user interfaces**.  
React was initially designed for building **single-page web applications (SPAs)** and to simplify the development process by allowing developers to write **reusable components** (blocks of code that can be used multiple times).

Over time, React has grown with additional packages that make building applications even simpler to run, understand, and maintain.  
For instance:

- **React-Bootstrap** – provides prebuilt styles to simplify CSS programming  
- **React-Router** – enables React to support multi-page applications that are faster and easier to navigate without page reloads  

🔗 **For a full tutorial:** [React Tutorial](https://react.dev/learn)

---

## How Does React Work?

React works in **declarative code**, which means the developer tells the application *what to do*, not *how to do it*.  
Declarative code describes what each element of the application should be.

For example:

```jsx
function App() {
  return (
    <>
      <HeroImage />
      <MissionStatement />
      <Team />
      <Footer />
    </>
  );
}
```
Developers use declarative code to create components that determine how information is displayed.
Each piece of code above is a component rendered to the virtual DOM. React knows what to render (HeroImage, MissionStatement, etc.), but not how; that’s for the developer to define.

Notice that the code looks like HTML. That’s because React uses JSX (JavaScript XML), which allows developers to write HTML-like syntax in JavaScript. JSX is then compiled into regular JavaScript using Babel, a JavaScript compiler that translates modern syntax into code browsers understand.

``It’s possible to use React without JSX or ES6 (removing the need for Babel), but it can make your codebase more complex.
React can also integrate with build systems like Gatsby and Next.js.``

More on React without ES6 and JSX

Different React Build Systems

## What Are React Components?

React components are reusable UI elements that allow developers to split applications into independent, modular sections of code.
Each component acts independently but can interact via props (inputs that carry data) to return a React element, which is what appears on screen.

### State vs Props

**State:** Represents the app’s “snapshot” at any given time. It’s internal to a component and managed directly within it.

**Props:** Variables passed into a component from a parent component. Props make components reusable by providing external data without changing the component’s internal logic.

Think of props like a global variable (with limits) and state like a local variable.

This structure gives developers greater control over data flow and rendering, allowing specific responsibilities to belong to particular components.

You can learn more about State vs Props in React.

## Starting a React Application

Creating your first React app is simple! You’ll need Node.js installed to use the create-react-app command.

### Steps

You can just open your terminal and navigate to the directory where you want to create your app.

Run the following command:

```jsx
npx create-react-app my-react-app
```
This command installs all the necessary packages to run a React application.

Once it finishes, you’ll see a “Happy hacking!” message.
Then, move into your new app directory.

```jsx
cd my-react-app
```
Start your development server:
```jsx
npm start
```
Your app should automatically open in your browser at http://localhost:3000.

If not, open it manually, and you’ll see the default React welcome page.

## Class-Based vs Functional Components

React supports two main types of components: class-based and functional.

### Class-Based Components

The original way to write React components.

They are **stateful**, manage their own state, and use lifecycle methods.

```jsx
class Welcome extends React.Component {
  constructor(props) {
    super(props);
    this.state = { message: 'Hello, World!' };
  }

  render() {
    return <h1>{this.state.message}</h1>;
  }
}
```

### Functional Components

The modern, more straightforward way to create components using functions and React Hooks.

With arrow function:

```jsx
const Welcome = () => {
  const [message, setMessage] = useState('Hello, World!');
  return <h1>{message}</h1>;
};
```
Without arrow function:

```jsx
function Welcome() {
  const [message, setMessage] = useState('Hello, World!');
  return <h1>{message}</h1>;
}
```

Functional components are more concise, easier to test, and leverage hooks like:

```useState()``` – manages state

```useEffect()``` – manages side effects

These hooks replace many class-based lifecycle methods, making React development more intuitive.

## React Packages

React’s ecosystem thrives on npm packages, which save time and increase productivity.
They help handle tasks like testing, animations, and API calls.

Here are some popular ones:

| Package                       | Description                                |
| ----------------------------- | ------------------------------------------ |
| **Axios**                     | Promise-based HTTP client for API requests |
| **React Router DOM**          | Enables multi-page navigation              |
| **React Responsive Carousel** | Responsive carousel component              |
| **React Awesome Button**      | Stylish, prebuilt button component         |
| **React Calendar**            | Calendar component for React               |
| **React Testing Library**     | Lightweight testing utility                |
| **Framer Motion**             | Animation library for React                |

It’s nearly impossible to build a complete React application without at least one npm package.
These libraries give your app that extra oomph while improving developer experience.

List of Useful npm Packages for React Developers

14 Useful Packages Every React Developer Should Know

## Learn More About React

This guide offers a simple introduction to React and how to get started.
For more comprehensive tutorials, check out:

[React.js Official Docs](https://react.dev/)

[React Tutorial](https://react.dev/learn)

[Create a New React App](https://react.dev/learn/start-a-new-react-project)

**Tip:** Everything in React is a component. Keep things small, reusable, and declarative.
React will handle the rest.
