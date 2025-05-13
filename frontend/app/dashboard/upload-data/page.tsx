
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUploader } from "@/components/features/upload-data/file-uploader";

export default function UploadDataPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Upload Your Email Data</CardTitle>
          <CardDescription>
            Drag and drop your email data files (e.g., .mbox, .pst, .eml folders) or click to select files.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FileUploader />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Processing Queue</CardTitle>
          <CardDescription>
            View the status of your uploaded files.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Placeholder for file processing queue */}
          <p className="text-sm text-muted-foreground">
            No files currently processing. Uploaded files will appear here with their status.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
