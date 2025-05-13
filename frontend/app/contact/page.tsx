
"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useForm, type SubmitHandler } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import Link from "next/link";
import { MailThreaderLogo } from "@/components/icons/mail-threader-logo";
import { MailIcon, MapPinIcon, PhoneIcon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";


const contactSchema = z.object({
  name: z.string().min(2, { message: "Name must be at least 2 characters." }).max(100),
  email: z.string().email({ message: "Please enter a valid email address." }),
  subject: z.string().min(5, { message: "Subject must be at least 5 characters."}).max(150),
  message: z.string().min(10, { message: "Message must be at least 10 characters." }).max(1000),
});

type ContactFormInputs = z.infer<typeof contactSchema>;

export default function ContactPage() {
  const currentYear = new Date().getFullYear();
  const { toast } = useToast();

  const form = useForm<ContactFormInputs>({
    resolver: zodResolver(contactSchema),
    defaultValues: {
      name: "",
      email: "",
      subject: "",
      message: "",
    },
  });

  const onSubmit: SubmitHandler<ContactFormInputs> = async (data) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log("Contact form submitted:", data);

    toast({
      title: "Message Sent!",
      description: "Thank you for reaching out. We'll get back to you soon.",
      variant: "default",
    });
    form.reset();
  };

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex items-center gap-2 mr-6">
            <MailThreaderLogo className="h-7 w-7 text-primary" />
            <span className="text-xl font-semibold">Mail-Threader</span>
          </Link>
          <nav className="flex items-center space-x-2 sm:space-x-4 text-sm font-medium ml-auto">
            <Button variant="ghost" asChild>
              <Link href="/about">About</Link>
            </Button>
            <Button variant="ghost" asChild>
              <Link href="/contact">Contact</Link>
            </Button>
            <Button variant="ghost" asChild>
              <Link href="/login">Login</Link>
            </Button>
            <Button asChild>
              <Link href="/signup">Sign Up</Link>
            </Button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 py-12 md:py-16 lg:py-20">
        <div className="container max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <Card className="shadow-xl overflow-hidden">
            <CardHeader className="bg-muted/30 p-8 md:p-12 text-center">
               <div className="inline-flex items-center justify-center p-4 bg-primary/10 rounded-full mb-6 mx-auto">
                 <MailThreaderLogo className="h-16 w-16 text-primary" />
              </div>
              <CardTitle className="text-4xl md:text-5xl font-bold tracking-tight">Get in Touch</CardTitle>
              <CardDescription className="mt-4 text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
                We&apos;d love to hear from you. Reach out with any questions, feedback, or inquiries.
              </CardDescription>
            </CardHeader>

            <div className="grid md:grid-cols-2">
              <CardContent className="p-8 space-y-8 bg-card">
                <h3 className="text-2xl font-semibold text-primary border-b pb-3">Contact Information</h3>
                <div className="space-y-6 text-foreground/90">
                  <div className="flex items-start gap-4">
                    <MailIcon className="h-7 w-7 text-primary flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-lg">Email Us</h4>
                      <a href="mailto:support@mailthreader.com" className="text-muted-foreground hover:text-primary hover:underline">support@mailthreader.com</a>
                      <p className="text-sm text-muted-foreground/80">For support and general inquiries.</p>
                    </div>
                  </div>
                   <div className="flex items-start gap-4">
                    <PhoneIcon className="h-7 w-7 text-primary flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-lg">Call Us</h4>
                      <p className="text-muted-foreground">+1 (555) 123-4567</p>
                      <p className="text-sm text-muted-foreground/80">Mon-Fri, 9am - 5pm EST.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <MapPinIcon className="h-7 w-7 text-primary flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-lg">Our Office</h4>
                      <p className="text-muted-foreground">123 AI Street, Innovation City, Techtopia 12345</p>
                       <p className="text-sm text-muted-foreground/80">Visits by appointment only.</p>
                    </div>
                  </div>
                </div>
              </CardContent>

              <CardContent className="p-8 border-t md:border-t-0 md:border-l bg-card">
                <h3 className="text-2xl font-semibold text-primary mb-6 border-b pb-3">Send Us a Message</h3>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                    <FormField
                      control={form.control}
                      name="name"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Full Name</FormLabel>
                          <FormControl>
                            <Input placeholder="e.g., Jane Doe" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="email"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Email Address</FormLabel>
                          <FormControl>
                            <Input type="email" placeholder="you@example.com" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="subject"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Subject</FormLabel>
                          <FormControl>
                            <Input placeholder="e.g., Inquiry about Enterprise Plan" {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="message"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Your Message</FormLabel>
                          <FormControl>
                            <Textarea placeholder="Please type your message here..." rows={5} {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <Button type="submit" className="w-full text-lg py-6" disabled={form.formState.isSubmitting}>
                      {form.formState.isSubmitting ? "Sending..." : "Send Message"}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </div>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 border-t bg-background">
        <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-muted-foreground">
          <div className="flex justify-center space-x-6 mb-4">
              <Link href="/about" className="text-sm hover:text-primary hover:underline">About Us</Link>
              <Link href="/contact" className="text-sm hover:text-primary hover:underline">Contact</Link>
              <Link href="/#features" className="text-sm hover:text-primary hover:underline">Features</Link>
          </div>
          <p>&copy; {currentYear} Mail-Threader. All rights reserved.</p>
          <p className="text-sm mt-2">Unlocking understanding from complex email data.</p>
        </div>
      </footer>
    </div>
  );
}
